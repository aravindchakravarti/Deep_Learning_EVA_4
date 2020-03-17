def runTheModel (model, device, optimizer, train_loader, test_loader, train, test, epochs = 20):
	for epoch in range(epochs):
		print("EPOCH:", epoch)
		train(model, device, train_loader, optimizer, epoch)
		# scheduler.step()
		test(model, device, test_loader)