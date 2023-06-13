import sys
from pstats import Stats, SortKey

if __name__ == '__main__':
    pass
    p = Stats(sys.argv[1])
    n = 10
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(n)
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(n)
