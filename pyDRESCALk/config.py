def init(arg):
    """ Global variables declaration here. The variables declared within this function in this file are shared across other files and functions during import."""
    global time, flag, flag_flops, flag_memory, totalflops, memoryAlloc, precision
    time = {}
    totalflops = {}
    memoryAlloc = {}
    flag =0
    flag_flops=0
    flag_memory=0
