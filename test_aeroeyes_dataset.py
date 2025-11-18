"""
Test script for AeroEyes Dataset
Run this to verify dataset implementation works correctly
"""
import sys
import os

# Add rfdetr to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from rfdetr.datasets.aeroeyes import AeroEyesDataset, aeroeyes_collate_fn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    print("✅ Successfully imported AeroEyesDataset")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_dataset_creation():
    """Test dataset initialization"""
    print("\n" + "="*50)
    print("Test 1: Dataset Creation")
    print("="*50)
    
    # Update this path to your dataset location
    dataset_path = "dataset/observing_train"
    
    if not os.path.exists(dataset_path):
        print(f"⚠️  Dataset path not found: {dataset_path}")
        print("   Please update dataset_path in this script")
        return False
    
    try:
        # Create transforms
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        dataset = AeroEyesDataset(
            root_dir=dataset_path,
            transforms=transform,
            max_frames_per_video=10,
            include_negatives=True
        )
        
        print(f"✅ Dataset created successfully!")
        print(f"   Dataset size: {len(dataset)}")
        return True, dataset
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_single_sample(dataset):
    """Test loading a single sample"""
    print("\n" + "="*50)
    print("Test 2: Single Sample Loading")
    print("="*50)
    
    if dataset is None or len(dataset) == 0:
        print("⚠️  Dataset is empty, skipping test")
        return False
    
    try:
        ref_imgs, frame, target = dataset[0]
        
        print(f"✅ Sample loaded successfully!")
        print(f"   Reference images: {len(ref_imgs)}")
        print(f"   Frame shape: {frame.shape}")
        print(f"   Target keys: {list(target.keys())}")
        print(f"   Boxes shape: {target['boxes'].shape}")
        print(f"   Labels shape: {target['labels'].shape}")
        print(f"   Image ID: {target['image_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader(dataset):
    """Test DataLoader with collate function"""
    print("\n" + "="*50)
    print("Test 3: DataLoader with Collate Function")
    print("="*50)
    
    if dataset is None or len(dataset) == 0:
        print("⚠️  Dataset is empty, skipping test")
        return False
    
    try:
        loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=aeroeyes_collate_fn,
            shuffle=False,
            num_workers=0  # Set to 0 for debugging
        )
        
        ref_imgs_batch, frames_batch, targets_batch = next(iter(loader))
        
        print(f"✅ DataLoader works correctly!")
        print(f"   Batch ref imgs: {len(ref_imgs_batch)} samples")
        print(f"   Each sample has {len(ref_imgs_batch[0])} ref images")
        print(f"   Frames batch shape: {frames_batch.tensors.shape}")
        print(f"   Frames mask shape: {frames_batch.mask.shape}")
        print(f"   Targets batch: {len(targets_batch)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*50)
    print("AeroEyes Dataset Test Suite")
    print("="*50)
    
    # Test 1: Dataset creation
    success, dataset = test_dataset_creation()
    if not success:
        print("\n❌ Dataset creation failed. Please check your dataset path and structure.")
        return
    
    # Test 2: Single sample
    if not test_single_sample(dataset):
        print("\n❌ Single sample loading failed.")
        return
    
    # Test 3: DataLoader
    if not test_dataloader(dataset):
        print("\n❌ DataLoader test failed.")
        return
    
    print("\n" + "="*50)
    print("✅ All tests passed!")
    print("="*50)

if __name__ == "__main__":
    main()

