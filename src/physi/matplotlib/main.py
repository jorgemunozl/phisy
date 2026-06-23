import torch 
import numpy as np
import os


class Plotty():
    """
    A helper to plot a lot of different kinds
    :var flag: Description
    :vartype flag: Literal[True]
    """
    def __init__(self, x_sample=1000, t_sample=1000):
        self.x_sample = torch.linspace(0, 1, x_sample).unsqueeze(1)
        self.t_sample = np.linspace(0, 1, t_sample)
        self.time_one = torch.ones((x_sample, 1))
        self.dir = "plots"
        os.makedirs(self.dir, exist_ok=True)

    def heat_comparation(self, model):

        directory = self.dir+"/analytic"
        os.makedirs(directory, exist_ok=True)
        file_name = "first.png"
        file_path = os.path.join(directory, file_name)

        with torch.no_grad():
            t_torch = torch.ones((1000, 1), dtype=torch.float)
            for i in self.t_sample:
                t_eval = t_torch * i
                y = model(self.x_sample, t_eval)
                y_ = heat_function(self.x_sample, i)
                if i == 0:
                    plt.plot(self.x_sample, y_, color='red',
                             label='True', linewidth=1)
                    plt.plot(self.x_sample, y, color='blue',
                             label='Predicted', linewidth=1)
                plt.plot(self.x_sample, y, color='blue', linewidth=1)
                plt.plot(self.x_sample, y_, color='red', linewidth=1)
        plt.title("Comparation")
        plt.legend()
        plt.savefig(file_path, dpi=600)

    def time_vs_error(self, model):

        directory = self.dir+"/error"
        os.makedirs(directory, exist_ok=True)
        file_name = "time_vs_error.png"
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, file_name)

        with torch.no_grad():
            time_one = torch.ones((len(self.x_sample), 1))
            error_t = np.zeros((len(self.t_sample)))
            cont = 0
            for i in self.t_sample:
                t_eval = time_one * i
                y_predict = model(self.x_sample, t_eval)
                y_true = heat_function(self.x_sample.squeeze(),
                                       t_eval.squeeze())
                E = error(y_true, y_predict)
                error_ = E.MAPE()
                error_ = torch.mean(error_)
                error_t[cont] = error_.numpy()
                cont += 1
        plt.plot(self.t_sample, error_t)
        plt.savefig(file_path, dpi=600)

    def three_dimensional(self):  # For oct 4
        fig, ax = plt.subplots(figsize=(8, 4), )
        X, Y = np.meshgrid(self.x_sample.numpy(), self.t_sample)

        T = heat_function(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, T, cmap='plasma')

        fig.colorbar(surf, ax=ax, shrink=0.5,
                     aspect=10, label='Temperature (°C)')
        ax.set_xlabel('x')
        ax.set_ylabel('Time')
        ax.set_zlabel('Temperature (°C)')
        ax.set_title('Heat Equation')
        plt.savefig("3d", dpi=600)

    def scalar(self):
        self.x_sample = self.x_sample.squeeze(1)
        X, Y = np.meshgrid(self.x_sample.numpy(), self.t_sample)
        T = heat_function(X, Y)
        plt.figure()
        plt.imshow(T, origin='lower', cmap='plasma', aspect='auto')
        plt.colorbar(label='Temperature')
        plt.xlabel('x')
        plt.ylabel('Time')
        plt.title('Heat equation one dimension')
        plt.savefig("heat")

    def animation_mape(self, model, epochs=200):
        directory = self.dir+"/animations"
        os.makedirs(directory, exist_ok=True)
        file_name = "animation_mape.mp4"
        file_path = os.path.join(directory, file_name)

        fig, ax = plt.subplots()
        line1, = ax.plot([], [], lw=2, label='MAPE')
        line2, = ax.plot([], [], lw=2, label='True')
        line3, = ax.plot([], [], lw=2, label='Predicted')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

        data1 = []
        data2 = []
        data3 = []

        for t in self.t_sample:
            y_true = heat_function(self.x_sample, t)

            with torch.no_grad():
                y_predict = model(self.x_sample, self.time_one*t)
            E = error(y_true, y_predict)

            data1.append(E.MAPE())
            data2.append(y_true)
            data3.append(y_predict)

        def update(frame):
            y1 = data1[frame]
            y2 = data2[frame]
            y3 = data3[frame]

            line1.set_data(self.x_sample, y1)
            line2.set_data(self.x_sample, y2)
            line3.set_data(self.x_sample, y3)
            return line1, line2, line3,

        animation_ = animation.FuncAnimation(
            fig, update, frames=100
        )
        animation_.save(file_path, writer='ffmpeg')

    def error_mape_fixed_t(self, model, t: float):

        directory = self.dir+"/error"
        os.makedirs(directory, exist_ok=True)
        number = f"{round(t, 2)}"
        file_name = f"error_mape_s_fixed_{number.replace(".", "")}.png"
        file_path = os.path.join(directory, file_name)
        t_eval = self.time_one * t

        with torch.no_grad():
            y_predict = model(self.x_sample, t_eval)

        y_true = heat_function(self.x_sample, t_eval)
        E = error(y_true, y_predict)
        error_mape_ = E.MAPE()

        plt.plot(self.x_sample.numpy(), error_mape_, label="MAPE Error")
        plt.plot(self.x_sample.numpy(), y_true, label="True",
                 linewidth=1)
        plt.plot(self.x_sample.numpy(), y_predict.numpy(),
                 label="Predicted", linewidth=1)
        plt.title(f"Error MAPE for time {t}")
        plt.legend()
        plt.savefig(file_path, dpi=600)

    def error_mse_fixed_t(self, model, t):

        directory = self.dir+"/error"
        os.makedirs(directory, exist_ok=True)
        file_name = f"error_mse_fixed_{int(t*10)}.png"
        file_path = os.path.join(directory, file_name)

        t_eval = self.time_one * t
        y_true = heat_function(self.x_sample, t)

        with torch.no_grad():
            y_predict = model(self.x_sample, t_eval)

        E = error(y_true, y_predict)
        error_mse_ = E.MSE()
        plt.plot(self.x_sample.numpy(), error_mse_, label="MSE")
        plt.plot(self.x_sample.numpy(), y_true, label="True", linewidth=1)
        plt.plot(self.x_sample.numpy(), y_predict.numpy(),
                 label="Predicted", linewidth=1)
        plt.legend()
        plt.savefig(file_path, dpi=700)

    def animate_snapshot(self, model, snap, frame, flag):
        """Plot when using epochs and difference, for that
        matter you use three dimensional tensors, training(phase),
        flag = True -> save animation, otherwise, save_data"""

        # Plot the true Red
        if flag:
            # Plot the static true.
            # Take the tensor and somehow it plot it and plot it directly
            # animation.save()
            pass
        else:
            tensor = []
            for t in self.t_sample:
                with torch.no_grad():
                    y_predict = model(self.x_sample, self.time_one*t)
            tensor.append(y_predict)
            snap[frame] = tensor