from multiprocessing import Process, Manager

def func(num, data_lst):
    data_lst.append(num*num)



if __name__ == '__main__':
    with Manager() as manager:
        # 创建共享列表
        data_lst = manager.list()

        # 启动多个子进程，每个子进程将计算结果添加到data_lst中
        processes = [Process(target=func, args=(i, data_lst)) for i in range(5)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # 输出结果
        print(list(data_lst))
