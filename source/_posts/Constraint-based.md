---
title: 基于约束的因果发现方法
date: 2025-05-29 17:18:09
tags: causal-inference
mathjax: true
---





## 一、PC（Peter-Clark）算法

PC算法（Peter-Clark算法）是一种经典的贝叶斯网络结构学习方法，核心目标是从观测数据中推断变量间的因果关系或条件独立关系，构建有向无环图（DAG，即贝叶斯网络）。

PC算法过程：

1.先将所有变量进行连接，从而获得一个完全无向图。

2.通过条件独立性（d-separation）测试删除多余边获得骨架图。

3.随后识别骨架图中的对撞结构（v-结构），可以确定部分边的方向。

4.在确定所有对撞结构之后，通过DAG图的性质和Meek规则尝试确定剩余边的方向。

{% asset_img constraint_flow.png constraint_flow %}

####  一、获得完全无向图

{% asset_img complete_undirected_graph.png constraint_flow %}

#### 二、 获得骨架图



{% asset_img skeleton_drawing.png skeleton_drawing %}



条件独立性常用的方法为**Finsher-z检验**。由于PC算法假定所有的变量服从**多元高斯分布**，那么我们可以通过计算**偏相关系数**来判断变量之间是否条件独立。

偏相关系数定义：



{% asset_img correlationcoefficient.png partial correlation coefficient.png %}



对于任意两个变量i和j，在排除h个其他变量（$$h\leq k-2,k$$可能是变量总数等相关量） 影响后的$h$阶偏相关系数。

该公式通过已有的低阶相关系数来计算当前阶的偏相关系数 ，体现了在控制特定数量变量后两变量的关联程度。

**FisherZ变换**

为了检验偏相关系数$\rho$是否为0，需要将其通过**Fisher Z**变换转化为正态分布：

{% asset_img FisherZ.png FisherZ transform.png %}



原假设：ρ = 0，给定条件*k*下，变量*i*和*j*之间不存在条件相关性，即条件独立

备择假设：ρ ≠ 0



#### 三、识别对撞结构

DAG图的构成主要分为三个结构：链式，叉式，对撞式

##### 链式：

{% asset_img chain.png 链式 %}

<p style="text-align:center">$$(X\perp Y)|Z \& X\not \perp Y$$</p>		



##### 叉式：

{% asset_img v-shape.png 叉式 %}

<p style="text-align:center">$$(X\perp Y)|Z \& X\not \perp Y$$</p>		



##### 对撞式：

{% asset_img Collision.png 对撞式 %}

<p style="text-align:center">$$(X\perp Y)|Z \& X\perp Y$$</p>		



上述三种结构，只有对撞式中，固定**变量Z**时，**变量X**和**变量Y**由独立变为不独立。由此性质，我们可以找出骨架图中的对撞式结构。

#### 四、确定剩余边的方向

#####  **Meek规则：**

###### **1.避免产生新的碰撞：**

{% asset_img Meek_1.png Meek_1 %}



###### **2. 避免成环：**

{% asset_img Meek_2.png Meek_2 %}





###### **3. 避免生成新的碰撞结构（2）**

{% asset_img Meek_3.png Meek_3 %}



如果上述图，将AB边识别为A <- B时，由于D->B,C->B已知，为了避免成环，那么AD和AC边可以定下方向，即为A<-D, A<-C。

{% asset_img Meek_3_2.png Meek_3_2 %}



该图出现了新的对撞结构D->A<-C

故AB边的方向为A->B



###### **4. 避免生成新的碰撞结构（3）**

{% asset_img Meek_4.png Meek_4 %}



如果上述图，将AD边识别为D -> A时，由于B->D已知，为了避免成环，那么AB边可以定下方向，即为A<-B



又由于B<-C ，为了避免成环，AC边可定下方向，即为A<-C

{% asset_img Meek_4_2.png Meek_4_2 %}





该图出现了新的对撞结构D->A<-C

故AD边的方向为A->D

到此，可以生成一个**部分有向无环图，PC算法结束。**



## 二、FCI算法

FCI算法是PC算法的延申，PC算法假设所有的因素都被观察到，没有混杂因素影响，FCI则考虑了存在混杂因素。并且FCI的结果不为DAG图，为PAG图（部分祖先图）。

FCI会识别出双箭头的情况，当出现双箭头时，表示变量之间出现混杂因子。



**FCI算法过程：**

1.先将所有变量进行连接，从而获得一个完全无向图。

2.通过条件独立性（d-separation）测试删除多余边获得骨架图。

3.随后识别骨架图中的对撞结构（v-结构），可以确定部分边的方向。

4.在确定所有对撞结构之后，通过Meek规则尝试确定剩余边的方向。



**注意：FCI在构建边时会分别判断一条边的两侧是否该出现箭头，并且边的两侧互不影响。**



**可能出现边的情况：**



###  **第一种情况：**

{% asset_img AtoB.png  AtoB %}

<p style="text-align:center">A是B的因</p>		

可能是直接原因也可能是包含其他可观测变量的间接原因。同时AB之间也可能存在不可观测的混杂因子。

消除可能： **B不是A的因**



###  **第二种情况**：



{% asset_img Unobservable_variables.png  Unobservable variables%}

<p style="text-align:center">存在不可观测变量（L）是A和B的共因。</p>		



从L到A或从L到B的因果路径上可能存在可观测变量。

消除可能： **A不是B的因，B也不是A的因**





###  **第三种情况**：



{% asset_img causalB.png  causalB%}

<p style="text-align:center">A是B的因，或存在不可观测变量是A和B的共因，或二者皆有</p>

消除可能： **B不是A的因** 



### 第四种情况：

{% asset_img AlinkB.png  AlinkB%}

<p>这种类型的边表示下面情况中的一种：<b>
    <br/>
   	(a) A是B的因<br />
	(b) B是A的因<br />
	(c) 存在不可观测变量是A和B的共因<br />
	(d) （a）和（c）同时发生<br />
	(e) （b）和（c）同时发生
    </b>
</p>



### 注：

**后续过程与PC算法过程一致,但是对边方向的生成有所区别**



**例如：{X,Y,Z}，且存在未观测变量U影响X和Y：**

假设真实的因果结构为：

{% asset_img Example.png  Example%}

其中，**L**是一个**潜在变量**（不可观测）



**PC算法：**

由于 **L** 不可见，PC 算法无法解释 **X** 与 **Y**之间的相关性，只能认为它们之间可能存在直接关系。可能得到如下图：

{% asset_img Example_2.png  Example_2%}

误导为 X 与 Y 有直接因果关系。

**FCI算法：**

FCI 可以检测到 **X**与 **Y**的相关性无法通过观测变量解释，因此使用 $X\leftrightarrow Y$表示它们之间**可能存在潜在公共原因**：

{% asset_img Example_3.png  Example_3%}

## 三、参考文献

https://www.cnblogs.com/bleu/p/18412938

https://geekdaxue.co/read/causality_zh/chapter02-introduction_to_PAG.md

https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/FCI.html

https://wiki.swarma.org/index.php/%E5%9B%A0%E6%9E%9C%E8%A1%A8%E5%BE%81%E5%AD%A6%E4%B9%A0%EF%BC%9A%E9%97%AE%E9%A2%98%EF%BC%8C%E6%96%B9%E6%B3%95%E5%92%8C%E5%BA%94%E7%94%A8



{% asset_img dog.jpg dog %}
