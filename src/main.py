
def main():
    import os

    # Folder where the files are located
    directory = "../datasets"  # change this as needed
    substring_to_remove = "functions_only_"

    for filename in os.listdir(directory):
        if substring_to_remove in filename:
            new_name = filename.replace(substring_to_remove, "")
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    main()