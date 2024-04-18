
def square_domain_mask(points2D, size=1.0):
    # bool mask for point inside the square
    return ~((points2D.abs() > size).sum(-1).bool())

def disk_domain_mask(points2D, radius=1):
    # bool mask for point inside the unit disk
    return ~(points2D.pow(2).sum(-1) > radius)
