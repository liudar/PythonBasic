
# It might take some time, if it is too long, try to reload it.
# Dataset definition
# class FoodDataset:
#     def __init__(self, paths, labels, mode):
#         # mode: 'train' or 'eval'
#
#         self.paths = paths
#         self.labels = labels
#         trainTransform = transforms.Compose([
#             transforms.Resize(size=(128, 128)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(15),
#             transforms.ToTensor(),
#         ])
#         evalTransform = transforms.Compose([
#             transforms.Resize(size=(128, 128)),
#             transforms.ToTensor(),
#         ])
#         self.transform = trainTransform if mode == 'train' else evalTransform
#
#     # pytorch dataset class
#     def __len__(self):
#         return len(self.paths)
#
#     def __getitem__(self, index):
#         X = Image.open(self.paths[index])
#         X = self.transform(X)
#         Y = self.labels[index]
#         return X, Y
#
#     # help to get images for visualizing
#     def getbatch(self, indices):
#         images = []
#         labels = []
#         for index in indices:
#             image, label = self.__getitem__(index)
#             images.append(image)
#             labels.append(label)
#         return torch.stack(images), torch.tensor(labels)

# # help to get data path and label
# def get_paths_labels(path):
#     def my_key(name):
#         return int(name.replace(".jpg" ,"").split("_")[1] ) +100000 0 *int(name.split("_")[0])
#     imgnames = os.listdir(path)
#     imgnames.sort(key=my_key)
#     imgpaths = []
#     labels = []
#     for name in imgnames:
#         imgpaths.append(os.path.join(path, name))
#         labels.append(int(name.split('_')[0]))
#     return imgpaths, labels
#
# train_paths, train_labels = get_paths_labels(args.dataset_dir)
#
# train_set = FoodDataset(train_paths, train_labels, mode='eval')

if __name__ == '__main__':
    a = 4
    b = 1 if a == 1 else 2
    print(b)