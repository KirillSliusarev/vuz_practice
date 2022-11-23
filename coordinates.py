
def ToInternal(dot):
    deltagloby = (55.836406 - 55.833351)  # y
    deltaglobx = (37.634269 - 37.629103)  # x
    iny = (504.5 * 2)
    inx = (483 + 483)  # x
    DotToDot = [deltagloby / iny, deltaglobx / inx]

    globalcords = [55.836406, 37.629103]
    internalcords = [-483, 504.5]

    a=[]
    for i in range(2):
        temp = (globalcords[i] - dot[i])/DotToDot[i]
        a.append(internalcords[1-i] - temp)
    a.reverse()
    return a

def ToGlobal(dot):
    deltagloby = (55.836406 - 55.833351)  # y
    deltaglobx = (37.634269 - 37.629103)  # x
    iny = (504.5 * 2)
    inx = (483 + 483)  # x
    DotToDot = [deltagloby / iny, deltaglobx / inx]

    globalcords = [55.836406, 37.629103]
    internalcords = [-483, 504.5]

    a=[]
    for i in range(2):
        temp = (internalcords[i] - dot[i]) * (DotToDot[1-i])
        a.append(globalcords[1-i] - (temp))
    a.reverse()
    return a
