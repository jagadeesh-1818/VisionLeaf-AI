import splitfolders

# The folder with your unsplit data
input_folder = "all_images/"

# The folder where you want the split data (train, val, test)
output_folder = "dataset/"

# Split with a ratio.
# To get 80% train, 10% validation, and 10% test, you can use a .8, .1, .1 ratio.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.8, .1, .1))

print("Dataset split successfully!")