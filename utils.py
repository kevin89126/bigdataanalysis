def handle_date(date):
    try:
        date = date.split('.')
        _y = int(date[0])
        _m = int(date[1])
        _d = int(date[2])
        if _m > 12 or _m < 1:
            raise
        if _d > 31 or _d < 1:
            raise
    except Exception as e:
        print('[ERROR] Invalid format, the correct one is "2020/1/12"')
        raise
    print(_y, _m, _d)
    return _y, _m, _d
