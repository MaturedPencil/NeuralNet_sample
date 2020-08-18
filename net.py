import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Net(Chain):

    def __init__(self, n_mid_units_1=256, n_mid_units_2 = 300,
                 n_mid_units_3 = 100, n_out=10):
        super(Net, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units_1)
            self.l2 = L.Linear(n_mid_units_1, n_out)
            #self.l3 = L.Linear(n_mid_units_2, n_mid_units_3)
            #self.l4 = L.Linear(n_mid_units_3, n_out)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        #h3 = F.relu(self.l3(h2))
        #h4 = F.softmax(self.l4(h3))
        return F.softmax(h2)
