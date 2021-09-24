import copy
import time
import torch


def standard(epochs, network, loader_function, optimizer, criterion,
             print_every=100, accuracy_function=None, phases=('train', 'val')):
    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    since = time.time()

    best_model_wts = copy.deepcopy(network.state_dict())
    best_loss = 1000.0

    verbose = not ('val' in phases and len(phases) == 1)

    for i in range(epochs):
        if verbose and (i % print_every == 0 or i == epochs - 1):
            print("==== Iter {0} =====".format(i))
        # Each epoch can have different phases. May be train and validation, or only one of them
        for phase in phases:
            if phase == 'train':
                network.train()  # Set model to training mode
            else:
                network.eval()  # Set model to evaluate mode

            running_loss = total_items = running_acc = 0
            # Turn on and off the gradient computations, when needed
            with torch.set_grad_enabled(phase == 'train'):
                for x, y in loader_function(phase):
                    items_count = len(y)

                    y_hat = network(x)
                    loss = criterion(y_hat, y)
                    running_loss += loss.item() * items_count
                    total_items += items_count
                    if accuracy_function:
                        running_acc += accuracy_function(y_hat, y)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

            epoch_loss = running_loss / total_items

            if verbose and (i % print_every == 0 or i == epochs - 1):
                print("{1} Avg. Loss: {0}".format(epoch_loss, phase.capitalize()))
                if accuracy_function:
                    print("{1} Avg. Accuracy: {0}".format(running_acc / total_items, phase.capitalize()))
                print()

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                if len(phases) > 1:
                    # to save the model: if we are on a validation phase, then we must check there is another
                    # phase (i.e. train). If validation is the only phase, then the model is only to query.
                    best_model_wts = copy.deepcopy(network.state_dict())
            history[phase + '_loss'].append(epoch_loss)
            if accuracy_function:
                history[phase + '_acc'].append(running_acc / total_items)

    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    network.load_state_dict(best_model_wts)
    return network, history
