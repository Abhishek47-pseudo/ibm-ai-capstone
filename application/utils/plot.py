import matplotlib.pyplot as plt

def ts_plot(x,y,figx=10, figy=10, s_scatter=5, alpha_scatter=0.8, alpha_plot=0.2, title=None):
    """
    plot time series data
    """
    plt.figure(figsize=(figx,figy))
    if title:
        plt.title(title)
    plt.scatter(x,y,s=s_scatter,alpha=alpha_scatter)
    plt.plot(x,y,alpha=alpha_plot)

def ts_plot_pred(x,y,x_pred,y_pred,figx=10, figy=10, s_scatter=5, alpha_scatter=0.8, alpha_plot=0.2, title=None):
    """
    plot time series data and predictions
    """
    plt.figure(figsize=(figx,figy))
    if title:
        plt.title(title)
    plt.scatter(x,y,s=s_scatter,alpha=alpha_scatter, label="True values")
    plt.plot(x,y,alpha=alpha_plot)

    plt.scatter(x_pred,y_pred,s=s_scatter,alpha=alpha_scatter, label="Predicted values")
    plt.legend()
    plt.plot(x_pred,y_pred,alpha=alpha_plot)
    