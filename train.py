import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from utils import data_loader_wind, data_loader_wind_nl
from models import wind_models
from tqdm import tqdm
import scipy.io as sio
from torchsummary import summary


class FlatLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def count_parameters(model):
    print("model_summary")
    print("Layer_name" + "\t" * 7 + "Number of Parameters")
    print("=" * 100)
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t" * 10)
    for i in layer_name:
        try:
            bias = (i.bias is not None)
        except:
            bias = False
        if not bias:
            param = model_parameters[j].numel() + model_parameters[j + 1].numel()
            j = j + 2
        else:
            param = model_parameters[j].numel()
            j = j + 1
        print(str(i) + "\t" * 3 + str(param))
        total_params += param
    print("=" * 100)
    print(f"Total Params:{total_params}")
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def loss_batch_scaled(model, loss_func, xb, yb, opt=None, scaler=None):
    loss = loss_func(model(xb)*scaler, yb*scaler, reduction="none")

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss, len(xb)


def test_model(data_folder, model, loss_func, dev="cpu", scaler=None):
    test_dl = data_loader_wind.get_test_loader(data_folder,
                                               batch_size=64,
                                               num_workers=4,
                                               pin_memory=True if dev == torch.device("cuda") else False)
    # Calc validation loss
    test_loss = torch.zeros(3, device=dev)
    test_num = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in tqdm(test_dl, desc="Test"):
            if scaler is None:
                loss, num = loss_batch(model, loss_func, xb.to(dev), yb.to(dev))
            else:
                loss, num = loss_batch_scaled(model, loss_func, xb.to(dev), yb.to(dev), scaler=scaler)
            test_loss += torch.sum(loss, dim=0)
            test_num += num
        test_loss /= test_num

    print(f"Test loss: {test_loss}")


def fit(epochs, model, loss_func, opt, train_dl, valid_dl,
        device=torch.device('cpu'), save_every: int = None, tensorboard: bool = False, earlystopping=None):
    if tensorboard:
        writer = SummaryWriter(comment=f"_wind_NL_1h_{model.__class__.__name__}")
    start_time = time.time()
    best_val_loss = 1e300
    earlystopping_counter = 0
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0
        total_num = 0
        # for i, (xb, yb) in enumerate(tqdm(train_dl, desc="Batches", leave=False)):
        for i, (xb, yb) in enumerate(train_dl):
            loss, num = loss_batch(model, loss_func, xb.to(device), yb.to(device), opt)
            train_loss += loss
            total_num += num
            # if i > 100:
            #     break
        train_loss /= total_num

        # Calc validation loss
        val_loss = 0.0
        val_num = 0
        model.eval()
        with torch.no_grad():
            # for xb, yb in tqdm(valid_dl, desc="Validation", leave=False):
            for xb, yb in valid_dl:
                loss, num = loss_batch(model, loss_func, xb.to(device), yb.to(device))
                val_loss += loss
                val_num += num
            val_loss /= val_num
        #     losses, nums = zip(
        #         *[loss_batch(model, loss_func, xb.to(device), yb.to(device)) for xb, yb in valid_dl]
        #     )
        # val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            torch.save({
                'model': model,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, f"models/checkpoints/best_val_loss_model_{model.__class__.__name__}.pt")
            best_val_loss = val_loss
            earlystopping_counter = 0

        else:
            if earlystopping is not None:
                earlystopping_counter += 1
                if earlystopping_counter >= earlystopping:
                    print(f"Stopping early --> val_loss has not decreased over {earlystopping} epochs")
                    break

        print(f"Epoch: {epoch:5d}, Time: {(time.time() - start_time) / 60:.3f} min, "
              f"Train_loss: {train_loss:2.10f}, Val_loss: {val_loss:2.10f}"
              f", Early stopping counter: {earlystopping_counter}/{earlystopping}" if earlystopping is not None else "")

        if tensorboard:
            # add to tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
        if save_every is not None:
            if epoch % save_every == 0:
                # save model
                torch.save({
                    'model': model,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, f"models/checkpoints/model_epoch_{epoch}.pt")


def train_wind_dk(data_folder, epochs=30, dev=torch.device("cpu"), earlystopping=None):
    print(f"Device: {dev}")
    forecast_time = int(data_folder[-5])*6
    train_dl, valid_dl = data_loader_wind.get_train_valid_loader(data_folder,
                                                                 batch_size=64,
                                                                 random_seed=1337,
                                                                 valid_size=0.1,
                                                                 shuffle=True,
                                                                 num_workers=16,
                                                                 pin_memory=True if dev == torch.device("cuda") else False)

    ### Model definition ###
    models_to_test = []
    models_to_test.append(wind_models.CNN2DWind_DK(in_channels=5, output_channels=3, feature_maps=32, hidden_neurons=128))  # 20,309 Params  # 46.115
    models_to_test.append(wind_models.CNN2DAttWind_DK(in_channels=5, output_channels=3, feature_maps=32, hidden_neurons=128))  # 20,497 Params # 47.059
    models_to_test.append(wind_models.CNNDS2DDeconvWind_DK(in_channels=5, output_channels=3, feature_maps=32, hidden_neurons=128))  # 27.974 Params
    models_to_test.append(wind_models.CNN3DWind_DK(in_channels=1, output_channels=3, feature_maps=10, hidden_neurons=128))  # 20.309 Params  # 54.749
    models_to_test.append(wind_models.MultidimConvNetwork(channels=5, height=4, width=4, output_channels=3, kernels_per_layer=32, hidden_neurons=128))  # 34.266 Params

    # loop here over the models
    for model in models_to_test:
        # print("Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        summary(model, (5, 4, 4), device="cpu")
        # Put the model on GPU
        model.to(dev)
        # Define optimizer
        lr = 0.001
        opt = optim.Adam(model.parameters(), lr=lr)
        # Loss function
        # loss_func = F.mse_loss
        loss_func = F.l1_loss

        #### Training ####
        tensorboard = False
        save_every = 10
        if tensorboard:
            writer = SummaryWriter(comment=f"_wind_DK_{forecast_time}h_{model.__class__.__name__}")
        start_time = time.time()
        best_val_loss = 1e300
        earlystopping_counter = 0
        for epoch in tqdm(range(epochs), desc="Epochs"):
            model.train()
            train_loss = 0.0
            total_num = 0
            # for i, (xb, yb) in enumerate(tqdm(train_dl, desc="Batches", leave=False)):
            for i, (xb, yb) in enumerate(train_dl):
                loss, num = loss_batch(model, loss_func, xb.to(dev), yb.to(dev), opt)
                train_loss += loss
                total_num += num
                # if i > 100:
                #     break
            train_loss /= total_num

            # Calc validation loss
            val_loss = 0.0
            val_num = 0
            model.eval()
            with torch.no_grad():
                # for xb, yb in tqdm(valid_dl, desc="Validation", leave=False):
                for xb, yb in valid_dl:
                    loss, num = loss_batch(model, loss_func, xb.to(dev), yb.to(dev))
                    val_loss += loss
                    val_num += num
                val_loss /= val_num

            # Save the model with the best validation loss
            if val_loss < best_val_loss:
                torch.save({
                    'model': model,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, f"models/checkpoints/best_val_loss_model_{model.__class__.__name__}.pt")
                best_val_loss = val_loss
                earlystopping_counter = 0

            else:
                if earlystopping is not None:
                    earlystopping_counter += 1
                    if earlystopping_counter >= earlystopping:
                        print(f"Stopping early --> val_loss has not decreased over {earlystopping} epochs")
                        break

            # print(f"Epoch: {epoch:5d}, Time: {(time.time() - start_time) / 60:.3f} min, "
            #       f"Train_loss: {train_loss:2.10f}, Val_loss: {val_loss:2.10f}"
            #       f", Early stopping counter: {earlystopping_counter}/{earlystopping}" if earlystopping is not None else "")

            if tensorboard:
                # add to tensorboard
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
            if save_every is not None:
                if epoch % save_every == 0:
                    # save model
                    torch.save({
                        'model': model,
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': train_loss
                    }, f"models/checkpoints/model_epoch_{epoch}.pt")

        ### Finished training ###
        # Save best model
        load_best_val_model = torch.load(f"models/checkpoints/best_val_loss_model_{model.__class__.__name__}.pt")
        torch.save({'model': load_best_val_model['model'],
                    'state_dict': load_best_val_model['state_dict']},
                   f"models/trained_models/wind_model_DK_{forecast_time}h_{model.__class__.__name__}.pt")


def train_wind_nl(data_folder, epochs, input_timesteps, prediction_timestep, dev=torch.device("cpu"), earlystopping=None):
    print(f"Device: {dev}")

    train_dl, valid_dl = data_loader_wind_nl.get_train_valid_loader(data_folder,
                                                                    input_timesteps=input_timesteps,
                                                                    prediction_timestep=prediction_timestep,
                                                                    CTF=True,
                                                                    batch_size=64,
                                                                    random_seed=1337,
                                                                    valid_size=0.1,
                                                                    shuffle=True,
                                                                    num_workers=16,
                                                                    pin_memory=True if dev == torch.device("cuda") else False)

    ### Model definition ###
    models_to_test = []
    models_to_test.append(wind_models.CNN2DWind_NL(in_channels=7, output_channels=7, feature_maps=32, hidden_neurons=128))
    models_to_test.append(wind_models.CNN2DAttWind_NL(in_channels=7, output_channels=7, feature_maps=32, hidden_neurons=128))
    models_to_test.append(wind_models.CNNDS2DDeconvWind_NL(in_channels=7, output_channels=7, feature_maps=32, hidden_neurons=128))
    models_to_test.append(wind_models.CNN3DWind_NL(in_channels=1, output_channels=7, feature_maps=10, hidden_neurons=128))
    models_to_test.append(wind_models.MultidimConvNetwork(channels=7, height=6, width=6, output_channels=7, kernels_per_layer=16, hidden_neurons=128))

    # loop here over the models
    for model in models_to_test:
        # print("Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        summary(model, (7, input_timesteps, 6), device="cpu")
        # Put the model on GPU
        model.to(dev)
        # Define optimizer
        lr = 0.001
        opt = optim.Adam(model.parameters(), lr=lr)
        # Loss function
        # loss_func = F.mse_loss
        loss_func = F.l1_loss
        #### Training ####
        tensorboard = False
        save_every = 10
        if tensorboard:
            writer = SummaryWriter(comment=f"_wind_NL_{prediction_timestep}h_{model.__class__.__name__}")
        start_time = time.time()
        best_val_loss = 1e300
        earlystopping_counter = 0
        for epoch in tqdm(range(epochs), desc="Epochs"):
            model.train()
            train_loss = 0.0
            total_num = 0
            # for i, (xb, yb) in enumerate(tqdm(train_dl, desc="Batches", leave=False)):
            for i, (xb, yb) in enumerate(train_dl):
                loss, num = loss_batch(model, loss_func, xb.to(dev), yb.to(dev), opt)
                train_loss += loss
                total_num += num
                # if i > 100:
                #     break
            train_loss /= total_num

            # Calc validation loss
            val_loss = 0.0
            val_num = 0
            model.eval()
            with torch.no_grad():
                # for xb, yb in tqdm(valid_dl, desc="Validation", leave=False):
                for xb, yb in valid_dl:
                    loss, num = loss_batch(model, loss_func, xb.to(dev), yb.to(dev))
                    val_loss += loss
                    val_num += num
                val_loss /= val_num

            # Save the model with the best validation loss
            if val_loss < best_val_loss:
                torch.save({
                    'model': model,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, f"models/checkpoints/best_val_loss_model_{model.__class__.__name__}.pt")
                best_val_loss = val_loss
                earlystopping_counter = 0

            else:
                if earlystopping is not None:
                    earlystopping_counter += 1
                    if earlystopping_counter >= earlystopping:
                        print(f"Stopping early --> val_loss has not decreased over {earlystopping} epochs")
                        break

            # print(f"Epoch: {epoch:5d}, Time: {(time.time() - start_time) / 60:.3f} min, "
            #       f"Train_loss: {train_loss:2.10f}, Val_loss: {val_loss:2.10f}"
            #       f", Early stopping counter: {earlystopping_counter}/{earlystopping}" if earlystopping is not None else "")

            if tensorboard:
                # add to tensorboard
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
            if save_every is not None:
                if epoch % save_every == 0:
                    # save model
                    torch.save({
                        'model': model,
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': train_loss
                    }, f"models/checkpoints/model_epoch_{epoch}.pt")

        ### Finished training ###
        # Save best model
        load_best_val_model = torch.load(f"models/checkpoints/best_val_loss_model_{model.__class__.__name__}.pt")
        torch.save({'model': load_best_val_model['model'],
                    'state_dict': load_best_val_model['state_dict']},
                   f"models/trained_models/wind_model_NL_{prediction_timestep}h_{model.__class__.__name__}.pt")


if __name__ == "__main__":
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # torch.backends.cudnn.benchmark = True
    folder = "data/"
    for t in [1, 2, 3, 4]:
        print("NL dataset. Step: ", t)
        data = "Wind_data_NL/dataset.pkl"
        train_wind_nl(folder+data, epochs=150, input_timesteps=6, prediction_timestep=t, dev=dev, earlystopping=20)
        print("DK dataset. Step: ", t)
        data = f"Wind_data/lag=4/step{t}.mat"
        train_wind_dk(folder+data, epochs=150, dev=dev, earlystopping=20)

    ### Test the newly trained model ###
    # load the model architecture and the weights
    # loaded = torch.load("models/wind_model.pt")

    # loaded = torch.load("models/checkpoints/best_val_loss_model_CNN3DWind.pt")
    # model = loaded["model"]
    # model.load_state_dict(loaded["state_dict"])
    # model.to(dev)
    # # get the scaler of the corresponding dataset
    # scaler = torch.as_tensor(sio.loadmat(f"{folder}Wind_data/lag=4/scale4.mat")["y_max_tr"], device=dev)
    # test_model(folder+data, model, F.l1_loss, dev, scaler=scaler)


