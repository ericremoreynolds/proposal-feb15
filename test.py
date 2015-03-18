import clr

from System import Array, Func
from csharp import ToyModel

def q(x):
    return "%s:%s{%s}" % (type(x), len(x), ",".join(map(str, x)))

qf = Func[Array[float], str](q)    
    
ToyModel.Test(qf)