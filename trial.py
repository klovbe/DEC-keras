import os

os.environ["LD_LIBRARY_PATH"] = "/home/xysmlx/anaconda3/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:" \
                                "/usr/local/cuda/lib64/libcublas.so.8.0"

print(os.environ.get("LD_LIBRARY_PATH"))

class A(object):
    def __init__(self, xing, gender):
        self.namea = "aaa"
        self.xing = xing
        self.gender = gender

    def funca(self):
        print("function a : %s" % self.namea)


class B(A):
    def __init__(self, xing, gender, age):
        super(B, self).__init__(xing, age)
        self.nameb = "bbb"
        self.namea="ccc"
        self.xing = xing.upper()
        self.age = age + 1
        self.gender = gender.upper()

    def funcb(self):
        print("function b : %s" % self.nameb)



b = B("lin", "nan", 22)
print(b.nameb)

print(b.namea)

print(b.xing)

print(b.age)

b.funcb()
b.funca()
print(b.gender)
 #####