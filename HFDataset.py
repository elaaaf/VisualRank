from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class HFDataset(Dataset):

    def __init__(self, hf_dataset, split='train'):
        self.dataset = hf_dataset[split]
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize((224, 224)),
            transforms.ToTensor(), # Convert PIL to pytorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], #ImageNet
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            # Get image data
            img_data = self.dataset[idx]['image']
            
            # Apply transforms
            image = self.transform(img_data)
            return image, idx
        except Exception as e:
            print(f"Error loading image #{idx}: {str(e)}")

    def show_image(self, idx):

        image = self.dataset[idx]['image']
        
        # Display
        plt.figure(figsize=(3, 3))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
