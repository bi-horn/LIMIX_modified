#Dieser Code definiert eine Funktion is_array(a), die prüft, ob das übergebene Objekt a ein NumPy-Array ist

def is_array(a):
    pkg = a.__class__.__module__.split(".")[0]
    name = a.__class__.__name__

    return pkg == "numpy" and name == "ndarray"
