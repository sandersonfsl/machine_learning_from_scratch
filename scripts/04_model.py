from machine_learning_from_scratch.data import x
from machine_learning_from_scratch.model import predict

# Try two different choices of w and b on the first three deliveries
print("distances:      ", x[:3])
print("w=3, b=12:      ", predict(x[:3], w=3, b=12))
print("w=1, b=5:       ", predict(x[:3], w=1, b=5))
