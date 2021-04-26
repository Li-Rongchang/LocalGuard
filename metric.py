import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def classifier(args, label):

    pp0 = []
    pp1 = []
    pp2 = []
    pp3 = []

    for e in [10,20,30,40,50,60,70,80,90,100]:

        print('the e is {}'.format(e))
        aaa = np.load('./{}_epoch{}_emb2.npy'.format(args.dataset,e))
        y_true = label

        x_train, x_test, y_train, y_test = train_test_split(aaa, y_true, test_size=0.4, shuffle=False)

        clf0 = MLPClassifier(random_state=10, max_iter=500)
        clf0.fit(x_train, y_train)
        p0 = clf0.score(x_test, y_test)
        p0 = round(p0, 3)
        pp0.append(p0)

        # clf1 = LogisticRegression(random_state=10, max_iter=200)
        # clf1.fit(x_train, y_train)
        # p1 = clf1.score(x_test, y_test)
        #
        # clf2 = svm.SVC(random_state=10, max_iter=200)
        # clf2.fit(x_train, y_train)
        # p2 = clf2.score(x_test, y_test)
        #
        # clf3 = GradientBoostingClassifier(random_state=10)
        # clf3.fit(x_train, y_train)
        # p3 = clf3.score(x_test, y_test)
        #
        # pp1.append(p1)
        # pp2.append(p2)
        # pp3.append(p3)

    # np.save('./{}_leakage_mlp_gin.npy'.format(args.dataset), pp0)
    # print('mlp',pp0,'lr',pp1,'svm',pp2,'gb',pp3)
    print('mlp', pp0)
    total = 0
    for ele in range(0, len(pp0)):
        total = total + pp0[ele]
    print('mlp_ave',total/len(pp0))


def classifier_other(args, label):

    pp1 = []
    pp2 = []
    pp3 = []

    for e in [10,20,30,40,50,60,70,80,90,100]:

        print('the e is {}'.format(e))
        aaa = np.load('./{}_epoch{}_emb2.npy'.format(args.dataset,e))
        y_true = label

        x_train, x_test, y_train, y_test = train_test_split(aaa, y_true, test_size=0.6, shuffle=False)

        # clf0 = MLPClassifier(random_state=10, max_iter=200)
        # clf0.fit(x_train, y_train)
        # p0 = clf0.score(x_test, y_test)
        # p0 = round(p0, 3)
        # pp0.append(p0)

        clf1 = LogisticRegression(random_state=10, max_iter=200)
        clf1.fit(x_train, y_train)
        p1 = clf1.score(x_test, y_test)

        clf2 = svm.SVC(random_state=10, max_iter=200)
        clf2.fit(x_train, y_train)
        p2 = clf2.score(x_test, y_test)

        clf3 = GradientBoostingClassifier(random_state=10)
        clf3.fit(x_train, y_train)
        p3 = clf3.score(x_test, y_test)
        p1 = round(p1, 3)
        p2 = round(p2, 3)
        p3 = round(p3, 3)

        pp1.append(p1)
        pp2.append(p2)
        pp3.append(p3)

    # np.save('./{}_leakage_lr.npy'.format(args.dataset), pp1)
    # np.save('./{}_leakage_svm.npy'.format(args.dataset), pp2)
    # np.save('./{}_leakage_gb.npy'.format(args.dataset), pp3)

    print('lr',pp1)
    print('svm',pp2)
    print('gb',pp3)
    # print('mlp', pp0)
    total = 0
    for ele in range(0, len(pp1)):
        total = total + pp1[ele]
    print('lr_ave', total / len(pp1))

    total = 0
    for ele in range(0, len(pp2)):
        total = total + pp2[ele]
    print('svm_ave', total / len(pp2))

    total = 0
    for ele in range(0, len(pp3)):
        total = total + pp3[ele]
    print('gb_ave', total / len(pp3))

def ml_privacy(args, label):
    import ite
    co = ite.cost.MIShannon_DKL()
    ml = {}
    time = 5
    for i in range(time):
        ml.update({i:[]})
        for e in [10,20,30,40,50,60,70,80,90,100]:  #[10,20,30,40,50,60,70,80,90,100]

            print('the e is {}'.format(e))
            a = np.load('./{}_epoch{}_emb2.npy'.format(args.dataset,e))
            d = label.reshape((label.shape[0], 1))

            ds = [a.shape[1], 1]
            y = np.concatenate((a, d), axis=1)

            ii = co.estimation(y, ds)
            ii = round(ii, 3)

            ml[i].append(ii)

    tem = np.zeros(10)  # 验证隐私epoch的数量
    for v in range(time):  # 随机次数
        # print(ml[v])
        tem += np.array(ml[v])
    mean = {'ave':tem/time}
    # print(mean['ave'])


    np.save('./{}_leakage_ml.npy'.format(args.dataset), mean['ave'])
    print('ave', mean['ave'])