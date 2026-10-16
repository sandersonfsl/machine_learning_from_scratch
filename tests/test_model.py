from machine_learning_from_scratch.model import predict


def test_predict_applies_the_line():
    # With w=3 and b=12, a 2 km delivery takes 3 * 2 + 12 = 18 minutes
    assert predict(2.0, w=3, b=12) == 18.0
