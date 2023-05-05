import os, sys, json
import math
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torchvision.models import resnet50
import torch.optim.lr_scheduler as lr_scheduler


from utils import read_split_data, evaluate, Test_A_Dataset, MyDataSet, show_confMat






def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file = sys.stdout, colour = "green")
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim = 1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))

        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("-------Training with args-------\n", args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    tb_writer = SummaryWriter(log_dir = "./ResNet_runs")
    if os.path.exists(args.trained_weights) is False:  os.makedirs(args.trained_weights)

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.test_data_path,
                                                                                               args.test_data_path.split(
                                                                                                   "\\")[-1])

    img_size = [300, 384]  # train_size, val_size

    data_transform = {
        "train": transforms.Compose([transforms.Resize([img_size[0], img_size[0]]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize([img_size[1], img_size[1]]),
                                   transforms.CenterCrop(img_size[1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path = train_images_path,
                              images_class = train_images_label,
                              transform = data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path = val_images_path,
                            images_class = val_images_label,
                            transform = data_transform["val"])

    batch_size = args.batch_size

    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 0
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               pin_memory = True,
                                               num_workers = nw,
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size = batch_size,
                                             shuffle = False,
                                             pin_memory = True,
                                             num_workers = nw,
                                             )

    model = resnet50(pretrained = True).cuda()
    model.requires_grad_(True)
    l = []
    [l.append(name) for name, param in model.named_parameters()]
    last_layer = l[-1]
    # add classifier to the pre-trained network
    if "classifier" in last_layer:
        model.classifier = nn.Linear(2048, args.num_classes)
    elif "fc" in last_layer:
        model.fc = nn.Linear(2048, args.num_classes)
    model.to(device)

    # 如果--resume则载入last，--pretrained_weights存在则载入预训练权重
    if os.path.exists(args.pretrained_weights):
        weights_dict = torch.load(args.pretrained_weights, map_location = device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        print("------initial with resumed last-------")
        print(model.load_state_dict(load_weights_dict, strict = False))
    else:
        print("not found pretrained weights file: {}".format(args.pretrained_weights))

    if (args.resume_weight != "") and (args.resume == False):
        if os.path.exists(args.resume_weight):
            weights_dict = torch.load(args.resume_weight, map_location = device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print("------initial with pretrained IMAGNET-------")
            print(model.load_state_dict(load_weights_dict, strict = False))
        else:
            print("not found resume weights file: {}".format(args.resume_weight))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "fc" in name:
                para.requires_grad_(False)
            if "classifier" in name:
                para.requires_grad_(False)
            print(name, para.requires_grad)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr = args.lr, momentum = 0.9, weight_decay = 1E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lf)

    train_acc_record = []
    val_acc_record = []

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model = model,
                                                optimizer = optimizer,
                                                data_loader = train_loader,
                                                device = device,
                                                epoch = epoch)
        train_acc_record.append(train_acc)
        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model = model,
                                     data_loader = val_loader,
                                     device = device,
                                     epoch = epoch)
        val_acc_record.append(val_acc)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), os.path.join(args.trained_weights, "model_last.pth"))
        if val_acc >= max(val_acc_record):
            print("-----replace the best model-----")
            torch.save(model.state_dict(), os.path.join(args.trained_weights, "model_best.pth"))


def TestWholeDataset(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    classes_name = []
    with open("./Source Domain0_class_indices.json", 'r', encoding = 'utf-8') as file:
        injson = json.load(file)
    for item in injson:
        classes_name.append(item)
    print("-------Test the whole dataset with args-------\n", args)

    if os.path.exists(args.trained_weights) is False:  raise FileNotFoundError(
        "not found weights file: {}".format(args.trained_weights))

    img_size = [300, 384]  # train_size, val_size

    data_transform = transforms.Compose([transforms.Resize([img_size[1], img_size[1]]),
                                         transforms.CenterCrop(img_size[1]),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # 实例化数据集
    Dataset = datasets.ImageFolder(root = args.test_data_path,
                                   transform = data_transform)

    batch_size = args.batch_size

    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 0
    print('Using {} dataloader workers every process'.format(nw))

    dloader = torch.utils.data.DataLoader(Dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          pin_memory = True,
                                          num_workers = nw, )

    model = resnet50(pretrained = True).cuda()
    l = []
    [l.append(name) for name, param in model.named_parameters()]
    last_layer = l[-1]
    # add classifier to the pre-trained network
    if "classifier" in last_layer:
        model.classifier = nn.Linear(2048, args.num_classes)
    elif "fc" in last_layer:
        model.fc = nn.Linear(2048, args.num_classes)
    model.to(device)

    # 载入best训练权重
    if os.path.exists(os.path.join(args.trained_weights, "model_last.pth")):
        weights_dict = torch.load(args.trained_weights + "/model_last.pth", map_location = device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict = False)
        print("------loaded with best weight -------")

    conf_mat, acc = Test_A_Dataset(model, dloader, classes_name, args)
    show_confMat(conf_mat, classes_name, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type = int, default = 3)
    parser.add_argument('--epochs', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--lr', type = float, default = 0.01)
    parser.add_argument('--lrf', type = float, default = 0.01)

    parser.add_argument('--data_path', type = str,
                        default = "G:\\dataset\\")
    parser.add_argument('--test_data_path', type = str,
                        default = "D:\\")
    parser.add_argument('--model_type', type = str,
                        default = "ResNet50", help = "EfficientNetV2/ResNet50/WRN50-2/DenseNet161")
    parser.add_argument('--pretrained_weights', type = str, default = 'pre_efficientnetv2-s.pth',
                        help = 'initial weights path')

    parser.add_argument('--trained_weights', type = str,
                        default = 'G:\\',
                        help = 'trained weights path for test')

    parser.add_argument('--conf_Mat_dir', type = str,
                        default = 'G:\\',
                        help = 'dir to store conf_Mat')
    parser.add_argument('--resume', type = bool, default = False)
    parser.add_argument('--resume_weight', type = str,
                        default = 'G:\\',
                        help = 'resume the train')
    parser.add_argument('--freeze-layers', type = bool, default = False)
    parser.add_argument('--device', default = 'cuda:0', help = 'device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)
    TestWholeDataset(opt)
