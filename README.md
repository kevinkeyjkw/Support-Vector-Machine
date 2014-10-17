Support-Vector-Machine
======================
Instructions:
python3 a3main.py dataset.txt x
where dataset.txt is the data set you want to classify and find support vectors for
x is the value of C used in stochastic gradient ascent for finding alpha vector

This program is for linearly seperable data, I will be updating it to handle non-linearly seperable data sets later

Basically it uses stochastic gradient ascent to find the support vectors for the data. Support vectors are the points 
that have a corresponding positive alpha value. When c is low like the value 10, the margin(space between hyperplane and
closest points will be maximized. But when c is high like 1000, margin may be smaller, but slack will be minimized(there
are no misclassified classes such as the alienated circle in c=10 plot)


C=10 Hyperplane. Notice the lone circle with the triangles, this is a side-effect of a larger margin

![s1](https://raw.githubusercontent.com/kevinkeyjkw/Support-Vector-Machine/master/HyperplaneC=10.png)

C=1000 Hyperplane. Notice how the margin is smaller here, but the points are accurately classified into their respective classes

![s2](https://raw.githubusercontent.com/kevinkeyjkw/Support-Vector-Machine/master/HyperplanceC=1000.png)
