#Ein Dask-Array ist eine parallele Datenstruktur in Python, die entwickelt wurde, um große Datenmengen zu verarbeiten, die nicht mehr in den Hauptspeicher passen. Es ist Teil des Dask-Projekts, das darauf abzielt, parallele und verteilte Rechenfähigkeiten in Python bereitzustellen

def get_shape(x):
    from . import dask

    if dask.is_array(x) or dask.is_dataframe(x):
        return _get_dask_shape(x)
    return x.shape


def _get_dask_shape(x):
    import dask.array as da

    return da.compute(*x.shape)

# Diese Funktion nimmt ein Numpy-Array x entgegen und gibt eine 1D-Version von x zurück, wobei die Reihenfolge F (Fortran) verwendet wird. Sie wird verwendet, um ein 2D-Array in einen Vektor umzuwandeln
def vec(x):
    from numpy import reshape

    return reshape(x, (-1,) + x.shape[2:], order="F")


def unvec(x, shape):
    from numpy import reshape

    return reshape(x, shape, order="F")

#Diese Funktion führt ein elementweises Produkt (Hadamard-Produkt) zwischen zwei Arrays A und B durch. Sie verwendet die tile- und repeat-Funktionen aus NumPy, um die Operation auf beide Arrays anzuwenden und das Ergebnis zurückzugeben

def cdot(A, B):
    """
    𝙰⊙𝙱 = [𝙰₀𝙱₀ ... 𝙰₀𝙱ₙ 𝙰₁𝙱₀ ... 𝙰₁𝙱ₙ ... 𝙰ₘ𝙱ₙ].
    """
    from numpy import tile, repeat

    BB = tile(B, A.shape[1])
    AA = repeat(A, B.shape[1], axis=1)
    return AA * BB
