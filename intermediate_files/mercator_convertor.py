#coding:utf-8


EARTHRADIUS = 6370996.81
MCBAND = [12890594.86, 8362377.87, 5591021, 3481989.83, 1678043.12, 0]
LLBAND = [75, 60, 45, 30, 15, 0]
LL2MC = [
    (-0.00157021024440, 1.113207020616939e+005, 1.704480524535203e+015, -1.033898737604234e+016, 2.611266785660388e+016, -3.514966917665370e+016, 2.659570071840392e+016, -1.072501245418824e+016, 1.800819912950474e+015, 82.5), 
    (8.277824516172526e-004, 1.113207020463578e+005, 6.477955746671608e+008, -4.082003173641316e+009, 1.077490566351142e+010, -1.517187553151559e+010, 1.205306533862167e+010, -5.124939663577472e+009, 9.133119359512032e+008, 67.5), 
    (0.00337398766765, 1.113207020202162e+005, 4.481351045890365e+006, -2.339375119931662e+007, 7.968221547186455e+007, -1.159649932797253e+008, 9.723671115602145e+007, -4.366194633752821e+007, 8.477230501135234e+006, 52.5), 
    (0.00220636496208, 1.113207020209128e+005, 5.175186112841131e+004, 3.796837749470245e+006, 9.920137397791013e+005, -1.221952217112870e+006, 1.340652697009075e+006, -6.209436990984312e+005, 1.444169293806241e+005, 37.5), 
    (-3.441963504368392e-004, 1.113207020576856e+005, 2.782353980772752e+002, 2.485758690035394e+006, 6.070750963243378e+003, 5.482118345352118e+004, 9.540606633304236e+003, -2.710553267466450e+003, 1.405483844121726e+003, 22.5), 
    (-3.218135878613132e-004, 1.113207020701615e+005, 0.00369383431289, 8.237256402795718e+005, 0.46104986909093, 2.351343141331292e+003, 1.58060784298199, 8.77738589078284, 0.37238884252424, 7.45)
]
MC2LL = [
    (1.410526172116255e-008, 8.983055096488720e-006, -1.99398338163310, 2.009824383106796e+002, -1.872403703815547e+002, 91.60875166698430, -23.38765649603339, 2.57121317296198, -0.03801003308653, 1.733798120000000e+007), 
    (-7.435856389565537e-009, 8.983055097726239e-006, -0.78625201886289, 96.32687599759846, -1.85204757529826, -59.36935905485877, 47.40033549296737, -16.50741931063887, 2.28786674699375, 1.026014486000000e+007), 
    (-3.030883460898826e-008, 8.983055099835780e-006, 0.30071316287616, 59.74293618442277, 7.35798407487100, -25.38371002664745, 13.45380521110908, -3.29883767235584, 0.32710905363475, 6.856817370000000e+006), 
    (-1.981981304930552e-008, 8.983055099779535e-006, 0.03278182852591, 40.31678527705744, 0.65659298677277, -4.44255534477492, 0.85341911805263, 0.12923347998204, -0.04625736007561, 4.482777060000000e+006), 
    (3.091913710684370e-009, 8.983055096812155e-006, 0.00006995724062, 23.10934304144901, -0.00023663490511, -0.63218178102420, -0.00663494467273, 0.03430082397953, -0.00466043876332, 2.555164400000000e+006), 
    (2.890871144776878e-009, 8.983055095805407e-006, -0.00000003068298, 7.47137025468032, -0.00000353937994, -0.02145144861037, -0.00001234426596, 0.00010322952773, -0.00000323890364, 8.260885000000000e+005),
]


def get_loop(v, a, b):
    while v > b:
        v -= b - a
    while v < a:
        v += b - a
    return v

def get_range(v, a, b):
    if a:
        v = max(v, a)
    if b:
        v = min(v, b)
    return v

def convertor(lng, lat, factor):
    x = factor[0] + factor[1] * abs(lng)
    tmp = abs(lat) * 1.0 / factor[9]
    y = factor[2]
    for i in range(3, 9):
        y += factor[i] * pow(tmp, i-2)
    x = x if lng > 0 else -x
    y = y if lat > 0 else -y
    return x,y

def coord2mc(lng, lat):
    newlng = get_loop(lng, -180, 180)
    newlat = get_range(lat, -74, 74)
    factor = None
    for i, value in enumerate(LLBAND):
        if newlat >= value:
            factor = LL2MC[i]
            break
    if not factor:
        factor = LL2MC[-1]
    pointx, pointy = convertor(newlng, newlat, factor)
    return pointx, pointy

def mc2coord(pointx, pointy):
    factor = None
    for i, value in enumerate(MCBAND):
        if pointy >= value:
            factor = MC2LL[i]
            break
    lng, lat = convertor(pointx, pointy, factor)
    return round(lng, 6), round(lat, 6)

def pixel2coord(pointx, pointy, zoom=18):
    pointx = pointx * 1.0 / pow(2, zoom-18)
    pointy = pointy * 1.0 / pow(2, zoom-18)
    lng, lat = mc2coord(pointx, pointy)
    return lng, lat

def coord2pixel(lng, lat, zoom=18):
    pointx, pointy = coord2mc(lng, lat)
    pointx = pointx * pow(2, zoom-18)
    pointy = pointy * pow(2, zoom-18)
    return pointx, pointy


if __name__ == '__main__':
    import sys
    lng = float(sys.argv[1])
    lat = float(sys.argv[2])
    print lng, lat
    pointx, pointy = coord2pixel(lng, lat)
    print pointx, pointy
    newlng, newlat = pixel2coord(pointx, pointy)
    print newlng, newlat