def intersect(ival1, ival2):
    """
    return the intersection of two floating point intervals.
    :param ival1: [start, end]
    :param ival2: [start, end]
    :return: amount of overlap, or 0 if non-overlapping
    """
    st1, ed1 = ival1
    st2, ed2 = ival2
    if st1>=ed2 or st2>=ed1:
        return 0
    return min(ed1, ed2) - max(st1, st2)