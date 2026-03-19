from huggingface_hub import HfApi, create_repo
import os

def main():
    print("--- Hugging Face Dataset Uploader ---")
    
    # 1. Get Token
    print("Step 1: Authentication")
    print("If you have run 'huggingface-cli login', just press Enter.")
    token_input = input("Enter your Hugging Face Write Token (or press Enter): ").strip()
    
    token = token_input if token_input else None
    
    # 2. Verify Token & Get Username
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info['name']
        print(f"✅ Authenticated as: {username}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("Please check your token or run 'huggingface-cli login'.")
        return

    # 3. Configure Dataset Name
    print("\nStep 2: Dataset Configuration")
    default_name = f"{username}/urban-thermal-mapping"
    print(f"Suggested dataset name: {default_name}")
    
    dataset_name = input(f"Enter dataset name (or press Enter for '{default_name}'): ").strip()
    if not dataset_name:
        dataset_name = default_name
    
    # 4. Upload
    local_folder = "dataset"
    if not os.path.exists(local_folder):
        print(f"Error: Folder '{local_folder}' not found.")
        return

    print(f"\nStep 3: Uploading to '{dataset_name}'...")
    
    try:
        # Create repo if it doesn't exist
        print(f"Checking/Creating repository '{dataset_name}'...")
        api.create_repo(
            repo_id=dataset_name,
            repo_type="dataset",
            exist_ok=True,
            token=token
        )
        print(f"✅ Repository ready.")
        
        # Upload check
        files_count = len(os.listdir(local_folder))
        print(f"Found {files_count} items in '{local_folder}'.")
        
        # Strategy: Upload batch by batch
        # 1. Upload root files (metadata, txt files, etc.)
        print("\n--- Phase 1: Uploading root files ---")
        root_files = [f for f in os.listdir(local_folder) if os.path.isfile(os.path.join(local_folder, f))]
        for file in root_files:
            print(f"Uploading file: {file}")
            api.upload_file(
                path_or_fileobj=os.path.join(local_folder, file),
                path_in_repo=file,
                repo_id=dataset_name,
                repo_type="dataset"
            )
            
        # 2. Upload directories one by one (The Scenes)
        directories = [d for d in os.listdir(local_folder) if os.path.isdir(os.path.join(local_folder, d))]
        total_dirs = len(directories)
        print(f"\n--- Phase 2: Uploading {total_dirs} directories (batches) ---")
        
        for i, directory in enumerate(directories, 1):
            print(f"[{i}/{total_dirs}] Uploading folder: {directory} ...")
            try:
                api.upload_folder(
                    folder_path=os.path.join(local_folder, directory),
                    path_in_repo=directory,
                    repo_id=dataset_name,
                    repo_type="dataset",
                    # multi_commits=True,  # Assuming directory is small enough for single commit
                    # multi_commits_verbose=True
                )
                print(f"✅ Completed: {directory}")
            except Exception as e:
                print(f"⚠️ Failed to upload {directory}: {e}")
                print("Continuing to next folder...")

        print(f"\n🎉 Success! View your dataset here: https://huggingface.co/datasets/{dataset_name}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if "403" in str(e):
            print(f"Tip: You are logged in as '{username}'. You cannot create a dataset for 'prajnakudkuli' unless that is your username or organization.")

if __name__ == "__main__":
    main()
