import os

def rename_files_to_txt(root):
    # Get the list of all files and directories in the current directory
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    
    # Iterate through each file
    for file_name in files:
        # Skip files that already end with '.txt'
        if not file_name.endswith('.txt'):
            # Get the file's base name (without extension)
            base_name, _ = os.path.splitext(file_name)
            
            # Form the new file name with '.txt' extension
            new_name = base_name + '.txt'
            
            # Rename the file
            os.rename(os.path.join(root, file_name), os.path.join(root, new_name))
            print(f"Renamed '{file_name}' to '{new_name}'")

if __name__ == "__main__":
    rename_files_to_txt("out/yolox_dets_OnTheFly:0_each_sample2:0.8:LastFrame:0.7LenThresh:0RemUnconf:0.0LastNFrames:10MM:1sum_0.8InactPat:50DetConf:0.45NewTrackConf:0.6")