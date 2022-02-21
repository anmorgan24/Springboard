# self-defined functions


# Plot six random examples of images from each origin location

def plot_six_by_origin(cl):

    #get six random indices belonging to the selected class
    indx = train[train['domain'] == cl].sample(6, axis=0).image_name.values
    #print(indx)

    #plot
    fig, axes = plt.subplots(2,3, figsize=(12,6))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(PIL.Image.open(data_dir / indx[i]))
    fig.suptitle(cl, fontsize=18)
    plt.tight_layout()
    plt.show()



# Take dataframe and image_name for which we want bounding boxes and return list of x, y, w, h

def get_all_bboxes(df, image_name):
    image_bounding_boxes = df[df.image_name == image_name]

    bounding_boxes = []
    for _, row in image_bounding_boxes.iterrows():
        bounding_boxes.append((row.bbox_xmin, row.bbox_ymin, row.bbox_xmax-row.bbox_xmin, row.bbox_ymax-row.bbox_ymin))

    return bounding_boxes



# Plot examples of random images with their bounding boxes

def plot_image_examples(dataframe, rows = 3, cols = 3, title = 'Image examples', size = (10, 10)):
    fig, axs = plt.subplots(rows, cols, figsize=size)
    for row in range(rows):
        for col in range(cols):
            idx = np.random.randint(len(dataframe), size = 1)[0]
            img_id = dataframe.iloc[idx].image_name

            img = Image.open(data_dir / img_id)
            axs[row, col].imshow(img)

            bboxes = get_all_bboxes(dataframe, img_id)

            for bbox in bboxes:
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
                axs[row, col].add_patch(rect)

                axs[row, col].axis('off')
    plt.suptitle(title, fontsize = 18)


# Plot rugplot at top of plot

def upper_rugplot(data, height=.03, ax=None, **kwargs):
    from matplotlib.collections import LineCollection
    ax = ax or plt.gca()
    kwargs.setdefault("linewidth", 1)
    segs = np.stack((np.c_[data, data],
                     np.c_[np.ones_like(data), np.ones_like(data)-height]),
                    axis=-1)
    lc = LineCollection(segs, transform=ax.get_xaxis_transform(), **kwargs)
    ax.add_collection(lc)



# Plot single image with all its bounding boxes

def plot_image_with_bboxes(dataframe, img_name, title = 'Image examples', size = (5, 5)):
    fig, ax = plt.subplots(figsize=size)
    img = Image.open(data_dir / img_name)
    ax.imshow(img)

    bboxes = get_all_bboxes(dataframe, img_name)

    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        ax.axis('off')
    plt.suptitle(title, fontsize = 16)


# Return array of pixel brightness for image

def get_image_brightness(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get average brightness
    return np.array(gray).mean()


# Loop through dataframe, determine brightness array for each image, add
# corresponding brightness column to dataframe

def add_brightness_column(df):
    brightness = []
    for _, row in df.iterrows():
        img_id = row.image_name
        image = cv2.imread(str(data_dir / img_id))
        brightness.append(get_image_brightness(image))

    brightness_df = pd.DataFrame(brightness)
    brightness_df.columns = ['brightness']
    df = pd.concat([df, brightness_df], ignore_index=True, axis=1)
    df.columns = ['image_name', 'brightness']

    return df


# Return the percentage of pixels that have high green intensity

def get_percentage_of_green_pixels(image):
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get the green mask
    hsv_lower = (40, 40, 40)
    hsv_higher = (70, 255, 255)
    green_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)

    return float(np.sum(green_mask)) / 255 / (1024 * 1024)


# Return the percentage of pixels that have high yellow intensity

def get_percentage_of_yellow_pixels(image):
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get the green mask
    hsv_lower = (25, 40, 40)
    hsv_higher = (35, 255, 255)
    yellow_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)

    return float(np.sum(yellow_mask)) / 255 / (1024 * 1024)


# Loop through dataframe to determine green pixel intensity percentage
# per image and add green_pixel column

def add_green_pixels_percentage(df):
    green = []
    for _, row in df.iterrows():
        img_id = row.image_name
        image = cv2.imread(str(data_dir / img_id))
        green.append(get_percentage_of_green_pixels(image))

    green_df = pd.DataFrame(green)
    green_df.columns = ['green_pixels']
    df = pd.concat([df, green_df], ignore_index=True, axis=1)
    df.columns = ['image_name', 'green_pixels']

    return df


# Loop through dataframe to determine yellow intensity per image and add
# yellow_pixel column

def add_yellow_pixels_percentage(df):
    yellow = []
    for _, row in df.iterrows():
        img_id = row.image_name
        image = cv2.imread(str(data_dir / img_id))
        yellow.append(get_percentage_of_yellow_pixels(image))

    yellow_df = pd.DataFrame(yellow)
    yellow_df.columns = ['yellow_pixels']
    df = pd.concat([df, yellow_df], ignore_index=True, axis=1)
    df.columns = ['image_name', 'yellow_pixels']

    return df