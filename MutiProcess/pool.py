import time
from multiprocessing import Pool

"""
当我们创建进程池时，实际上是创建了多个子进程，并将它们放到一个可重用的进程池中。在这个进程池中，所有子进程都处于就绪状态，等待接收任务。
当我们向进程池提交任务时，进程池会自动选择一个可用的子进程来接收任务，
并将任务分配给该子进程执行。当子进程完成了它的任务后，它会重新回到进程池中，
继续等待下一个任务。这种方式可以避免不必要的进程创建和销毁，从而提高了程序的效率。
"""


# 定义回调函数打印结果
def callback_func(result):
    print("The result is:", result)


# 定义一个计算数字列表平方和的函数
def calc_sum(nums):
    print("in---")
    time.sleep(1)
    mysum = sum([x*x for x in nums])
    print("out---")
    return mysum

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 创建一个拥有4个进程的进程池
    p = Pool(processes = 2)

    """
        # map会使进程阻塞直到结果返回,因此下面一句会一直阻塞到所有计算结果返回
        # 通过进程池并行计算多个数字列表的平方和
        # result = p.map(calc_sum, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
        
        # 打印计算结果
        # print(result)
    """


    """
        # map_async不会阻塞,立即返回,返回的是一个MapResult对象，该对象可以用于检测和获取异步计算的结果
        result = p.map_async(calc_sum, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
        # 打印结果
        while not result.ready():
            print("Waiting for the result...")
            time.sleep(0.5)
    
        print("The final result is:", result.get())
    """

    # 待计算的数据列表
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]]

    # 使用共享变量存储计算结果
    manager = Manager()
    result_list = manager.list()

    # 遍历数据列表，异步提交任务
    for d in data:
        p.apply_async(calc_sum, args=(d, result_list))


    # # 也可以使用回调函数打印
    # for d in data:
    #     p.apply_async(calc_sum, args=(d, callback_func))


    # 关闭进程池
    p.close()
    p.join()

    # 打印结果
    print("The final results are:", result_list)



