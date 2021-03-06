__我想干什么？__

自己动手实现一遍大部分的机器学习的算法。

目前初步已经理解与推导了部分机器学习的相关公式，能够调用一定的sklearn API进行数据分析；但是仍然不满足，希望在公式推导之余，能够增加编程的活动，这样也不至于把推公式看成是一件枯燥无味的事情。

__期待实现什么目标？__
1. 如果比较顺利，是不是也可以写一个类似`sklearn`的开源库呢？哈哈，人不能没有梦想啊，万一实现了怎么办？
    这算是一个较高的目标吧。
2. 对大部分机器学习算法，__知其然知其所以然__ 。
3. 在代码实现的过程中参考`sklearn`等优秀开源库，提高代码能力。
    

__告诉自己：__
1. 我一直觉得机器学习或深度学习，是一种Application导向的任务，只有在实际运用、处理的过程中才能够形成对数据的感觉与Insight。
2. 虽然很多人说，编程只是工具的问题，也无所谓编程语言与框架的问题，但其在机器学习或深度学习过程中绝不可以轻视之。
    至少目前需要做到几点：
    1. 熟练掌握一门编程语言，目前而言毫无疑问就是Python；
    2. 熟练掌握一些开源框架，毫无疑问Sklearn、TensorFlow等的优先级较高；
        并不是其他开源框架如Caffe，Torch等并不好，至少结合理论部分，先干好一件事情，其他的会去学习的。
3. 没必要纠结要求自己的代码一步到位而苦思冥想地迟迟不敢动手编程，要有一种迭代优化的思想：先去做，发现问题，修改；发现问题，再修改 . . .

### 基本的文件管理
1. 所有代码存放在远程服务器上，位于`T450/scikit-learn-zero/`文件夹内
2. 并进行GitHub的代码托管，Repository的名字为：`scikit-learn-zero`
3. 以 `Logistic Regression` 算法为例，针对相关文件，给文件编写顺序与出命名的一般规则：
    1. `selfLogisticRegressionDoc.ipynb` 善加利用Jupyter Notebook。 文档部分为，对算法的理论与公式的推导——知其然知其所以然；后续代码部分为自己代码实现的检验。
    2. `learn-sklearnLogisticRegression.ipynb` ，在完成算法或模型的理论部分后，先利用sklearn的API进行数据的初步分析与API参数理解，是一个快速加深理论理解，并在部分调参过程中深入理解的过程。
    3. `selfLogisticRegression.py` 有关 Logistic Regression 相关的类或函数都将在该文件中进行代码实现，并在1中的Jupyter Notebook文件中进行检验。

#### 关于 `selfLogisticRegressionDoc.ipynb` 中的理论部分
1. 一定要自己动手推导公式与必要理论部分，这是最核心最基础的部分；
2. 将理论部分在白纸上认真整理；
3. 将白纸上的理论推导 __用打印机扫描的形式保存成电子档__；
    1. 一定要认真整理，条理清晰，甚至多写多思考几遍，也可当作公式的理解与记忆；
    1. 避免使用公式编辑器等工具浪费大量的不必要时间；
    2. 自己手写过程与内容，写作过程更流畅自由，更强化大脑深入理解；
    3. 自己的手写，回顾起来更亲切，同时能够联想出更丰富的东西；
    4. 将图片等格式形式引入到Markdown中更直接；
    5. 如有必要，再将电子版打印下来标记，也是非常好的过程

### 陆续进行以下算法的代码实现。。。
1. Logistic Regression
2. Linear Discrimination Analysis
3. Principal Component Analysis
4. . . . 