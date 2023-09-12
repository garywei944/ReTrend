def train(
        train_loader, val_loader, model, loss_fn, optimizer, epochs,
        device='cuda'
):
    best_f1 = 0
    for epoch in range(epochs):
        print('-' * 50)
        print(f'epoch {epoch + 1}')

        model.train()
        total_loss = 0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            running_loss = 0
            # Every data instance is an input + label pair
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            total_loss += loss.item()
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss
                print('  batch {} loss: {}'.format(i + 1, last_loss))

        total_loss = total_loss / len(train_loader)
        _val_loss, _pr, _re, _f1 = evaluate(val_loader, model, loss_fn)

        if _f1 > best_f1:
            best_f1 = _f1

        print(f'\ttraining loss: {total_loss} | val loss: {_val_loss}')
        print(f'\tf1: {_f1} | precision: {_pr} | recall: {_re}')

    print('-' * 50)
    print(f'Best f1 after {epochs} epochs: {best_f1}')
