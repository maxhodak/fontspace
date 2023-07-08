import h5py
import torch
from torch.utils.data import Dataset

class FontDataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        self.file = h5py.File(hdf5_file, 'r')
        self.data = self.file.get('fonts')[:]
        
        # normalize pixel values to 0-1
        self.data = self.data / 255.0
        self.transform = transform

    # Number of images in the dataset
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        single_font_images = self.data[index]
        single_font_images = torch.from_numpy(single_font_images).float()
        if self.transform:
            single_font_images = self.transform(single_font_images)
        return (index, single_font_images)
            
#         # Reshape into grid of images
        
#         # Reshape into individual images
#         single_font_images = single_font_images.reshape(-1, 1, 64, 64)

#         # Add 2 zero images to make the total 64
#         single_font_images = torch.cat([single_font_images, torch.zeros(2, 1, 64, 64)], dim=0)
        
#         # Now it can be reshaped into an 8x8 grid
#         grid_font_images = single_font_images.view(8, 8, 1, 64, 64)

#         return grid_font_images

def display_grid(images):
    # Reshape the tensor to 8x8 grid of 64x64 images
    grid = images.view(8, 8, 64, 64)
    
    # Convert the tensor to numpy array
    grid = grid.detach().numpy()
    
    # Combine the 8x8 grid of 64x64 images into a single 512x512 image
    grid_combined = np.block([[grid[i][j] for j in range(8)] for i in range(8)])

    plt.imshow(grid_combined)
    plt.axis('off')