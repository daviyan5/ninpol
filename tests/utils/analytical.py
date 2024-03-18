

def linear(centroid):
    return centroid[:, 0] + centroid[:, 1] + centroid[:, 2]

def quadratic(centroid):
    return centroid[:, 0]**2 + centroid[:, 1]**2 + centroid[:, 2]**2

def quarter_five_spot(centroid):
    return centroid[:, 0]