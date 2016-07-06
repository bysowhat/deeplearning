#coding:utf-8
import cPickle, gzip, numpy
import theano
import theano.tensor as T

# Load the dataset   b:binary二进制文件 无b则为字符  r:read w:write
f = gzip.open('db/mnist.pkl.gz', 'rb')
# 载入本地文件，回复python对象 cPickle用于对象序列化
# python不用显示声明变量
train_set, valid_set, test_set = cPickle.load(f)
#print train_set
'''
for v in train_set[1]:
    print v
'''
f.close()


#python 定义函数
def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    '''
    >>> a , b = (1,2)
    >>> print a , b
    1 2
    >>>
    '''
    #Python的元组与列表类似，不同之处在于元组的元素不能修改,元组使用小括号,列表使用方括号,元组创建很简单,只需要在括号中添加元素,并使用逗号隔开即可
    #numpy.asarray 将数组转为矩阵
    #floatX=float32 theano.config文件中配置的
    '''
    Shared区是供GPU、C代码使用的内存区，与Python的内存区独立，但是由Tensor变量联系着。
    Python的普通类型可以由theano.shared()方法转换。
    Shared变量更新可以用自己的set_value(),也可以在function的updates里面。
    另外提一句，Tensor里的变量，只有Shared类型的才有get_value()属性，支持直接查看值。
    '''
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    
    # T指theano
    return shared_x, T.cast(shared_y, 'int32')

test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)

batch_size = 500    # size of the minibatch

# accessing the third minibatch of the training set


print valid_set_x;
#打印matrix
print valid_set_x.get_value()

'''
[5,12,200,-2]
>>> z = x[1:3] # array "slicing":elements 1 through 3-1 = 2
>>> z
'''

data  = train_set_x[2 * batch_size: 3 * batch_size]
label = train_set_y[2 * batch_size: 3 * batch_size]