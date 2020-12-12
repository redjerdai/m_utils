#
import numpy
from matplotlib import pyplot, lines as mlines


def visualise_predictions(model, X_train, Y_train, X_test, Y_test, time_series=False, quantiles=(.05, .50, .95),
                          freq=.1):
    Y_train_hat = model.predict(X_train)
    Y_test_hat = model.predict(X_test)
    train_err, test_err = Y_train - Y_train_hat, Y_test - Y_test_hat
    if time_series:
        fig, ax = pyplot.subplots(3, 2, figsize=(10, 10))

        x_train, x_test = numpy.array(numpy.arange(Y_train.shape[0])), numpy.array(numpy.arange(Y_test.shape[0]))

        cum_train_true, cum_train_hat = numpy.cumprod(Y_train + 1), numpy.cumprod(Y_train_hat + 1)
        cum_test_true, cum_test_hat = numpy.cumprod(Y_test + 1), numpy.cumprod(Y_test_hat + 1)

        ax[0, 0].plot(x_train, cum_train_true, 'navy', x_train, cum_train_hat, 'blueviolet')
        true_train_line = mlines.Line2D([], [], color='navy', label='True Train')
        # marker='*', markersize=15, label='Blue stars')
        hat_train_line = mlines.Line2D([], [], color='blueviolet', label='Estimated Train')
        ax[0, 0].legend(handles=[true_train_line, hat_train_line])

        ax[0, 1].plot(x_test, cum_test_true, 'navy', x_test, cum_test_hat, 'blueviolet')
        true_test_line = mlines.Line2D([], [], color='navy', label='True Test')
        hat_test_line = mlines.Line2D([], [], color='blueviolet', label='Estimated Test')
        ax[0, 1].legend(handles=[true_test_line, hat_test_line])

        ax[1, 0].hist(train_err, 50, density=True, facecolor='dodgerblue', alpha=0.5)
        ax[1, 0].hist(test_err, 50, density=True, facecolor='aqua', alpha=0.5)
        train_err_line = mlines.Line2D([], [], color='dodgerblue', label='Train Errors')
        test_err_line = mlines.Line2D([], [], color='aqua', label='Test Errors')
        ax[1, 0].legend(handles=[train_err_line, test_err_line])

        train_err_ecdf__y, train_err_ecdf__x = numpy.histogram(train_err, bins=50, density=True)
        train_err_ecdf__y, train_err_ecdf__x = numpy.cumsum(train_err_ecdf__y), train_err_ecdf__x[:-1]
        train_err_ecdf__y = train_err_ecdf__y / numpy.max(train_err_ecdf__y)
        test_err_ecdf__y, test_err_ecdf__x = numpy.histogram(test_err, bins=50, density=True)
        test_err_ecdf__y, test_err_ecdf__x = numpy.cumsum(test_err_ecdf__y), test_err_ecdf__x[:-1]
        test_err_ecdf__y = test_err_ecdf__y / numpy.max(test_err_ecdf__y)

        ax[1, 1].plot(train_err_ecdf__x, train_err_ecdf__y, 'dodgerblue', test_err_ecdf__x, test_err_ecdf__y, 'aqua')
        train_err_line = mlines.Line2D([], [], color='dodgerblue', label='Train Errors')
        test_err_line = mlines.Line2D([], [], color='aqua', label='Test Errors')
        ax[1, 1].legend(handles=[train_err_line, test_err_line])

        # add here dynamic errors !
    else:
        raise Exception("Time Series functionality available only. Set 'time_series' parameter to True")
