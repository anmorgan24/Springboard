def plot_six_by_origin(cl):
    
    '''Plot six random examples of images from each origin location'''

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
 

    

def get_all_bboxes(df, image_name):
    
    '''Take dataframe and image_name for which we want bounding boxes and return list of x, y, w, h'''
    
    image_bounding_boxes = df[df.image_name == image_name]

    bounding_boxes = []
    for _, row in image_bounding_boxes.iterrows():
        bounding_boxes.append((row.bbox_xmin, row.bbox_ymin, row.bbox_xmax-row.bbox_xmin, row.bbox_ymax-row.bbox_ymin))

    return bounding_boxes




def plot_image_examples(dataframe, rows = 3, cols = 3, title = 'Image examples', size = (10, 10)):
    
    '''Plot examples of random images with their bounding boxes'''
    
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



    
def upper_rugplot(data, height=.03, ax=None, **kwargs):
    
    '''Plot rugplot at top of plot'''
    
    from matplotlib.collections import LineCollection
    ax = ax or plt.gca()
    kwargs.setdefault("linewidth", 1)
    segs = np.stack((np.c_[data, data],
                     np.c_[np.ones_like(data), np.ones_like(data)-height]),
                    axis=-1)
    lc = LineCollection(segs, transform=ax.get_xaxis_transform(), **kwargs)
    ax.add_collection(lc)




def plot_image_with_bboxes(dataframe, img_name, title = 'Image examples', size = (5, 5)):
    
    '''Plot single image with all its bounding boxes'''
    
    fig, ax = plt.subplots(figsize=size)
    img = Image.open(data_dir / img_name)
    ax.imshow(img)

    bboxes = get_all_bboxes(dataframe, img_name)

    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        ax.axis('off')
    plt.suptitle(title, fontsize = 16)




def get_image_brightness(image):
    
    '''Return array of pixel brightness for image'''
    
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get average brightness
    return np.array(gray).mean()




def add_brightness_column(df):
    
    '''Loop through dataframe, determine brightness array for each image, add corresponding brightness column to dataframe'''
    
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


 

def get_percentage_of_green_pixels(image):
    
    '''Return the percentage of pixels that have high green intensity'''
    
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get the green mask
    hsv_lower = (40, 40, 40)
    hsv_higher = (70, 255, 255)
    green_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)

    return float(np.sum(green_mask)) / 255 / (1024 * 1024)




def get_percentage_of_yellow_pixels(image):
    
    '''Return the percentage of pixels that have high yellow intensity'''
    
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get the green mask
    hsv_lower = (25, 40, 40)
    hsv_higher = (35, 255, 255)
    yellow_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)

    return float(np.sum(yellow_mask)) / 255 / (1024 * 1024)




def add_green_pixels_percentage(df):
    
    '''Loop through dataframe to determine green pixel intensity percentage per image and add green_pixel column'''
    
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




def add_yellow_pixels_percentage(df):
    
    '''Loop through dataframe to determine yellow intensity per image and add yellow_pixel column'''
    
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




def img2np(df, size = (64, 64)):
    
    '''Convert image set to set of arrays'''
    
    # iterating through each file
    for _, row in df.iterrows():
        img_id = row.image_name
        current_image = cv2.cvtColor(cv2.imread(str(data_dir / img_id)), cv2.COLOR_BGR2GRAY)

        # covert image to a matrix
        img_ts = np.squeeze(np.array(current_image))
        # turn that into a vector / 1D array
        img_ts = [img_ts.ravel()]
        try:
            # concatenate different images
            full_mat = np.concatenate((full_mat, img_ts))
        except UnboundLocalError:
            # if not assigned yet, assign one
            full_mat = img_ts
    return full_mat




def find_mean_img(full_mat, title, size = (1024, 1024)):
    
    '''Determine mean image of a set of images and plot'''
    
    # calculate the average
    mean_img = np.mean(full_mat, axis = 0)
    # reshape it back to a matrix
    mean_img = mean_img.reshape(size)
    plt.imshow(mean_img, vmin=0, vmax=255, cmap='Greys_r')
    plt.title(f'{title}')
    plt.axis('off')
    plt.show()
    return mean_img




def get_mean_of_yellow_pixels(image):
    
    '''Return mean value of all yellow pixel values per image'''
    
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get the green mask
    hsv_lower = (25, 40, 40)
    hsv_higher = (35, 255, 255)
    yellow_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)

    return float(np.mean(yellow_mask))




def get_std_of_yellow_pixels(image):
    
    '''Return standard deviation of all yellow pixel values per image'''
    
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get the green mask
    hsv_lower = (25, 40, 40)
    hsv_higher = (35, 255, 255)
    yellow_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)

    return float(np.std(yellow_mask))

 

    
def add_mean_yellow_pixels(df):
    
    '''Add mean of yellow pixel values to a given dataframe'''
    
    yellow = []
    for _, row in df.iterrows():
        img_id = row.image_name
        image = cv2.imread(str(data_dir / img_id))
        yellow.append(get_mean_of_yellow_pixels(image))

    yellow_df = pd.DataFrame(yellow)
    yellow_df.columns = ['yellow_pixel_mean']
    df = pd.concat([df, yellow_df], ignore_index=True, axis=1)
    df.columns = ['image_name', 'yellow_pixel_mean']

    return df

 

    
def add_std_yellow_pixels(df):
    
    '''Add standard deviation of yellow pixels per image to a given dataframe'''
    
    yellow = []
    for _, row in df.iterrows():
        img_id = row.image_name
        image = cv2.imread(str(data_dir / img_id))
        yellow.append(get_std_of_yellow_pixels(image))

    yellow_df = pd.DataFrame(yellow)
    yellow_df.columns = ['yellow_pixel_std']
    df = pd.concat([df, yellow_df], ignore_index=True, axis=1)
    df.columns = ['image_name', 'yellow_pixel_std']

    return df

 

    
def get_mean_of_green_pixels(image):
    
    '''Return mean value of all green pixel values per image'''
    
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get the green mask
    hsv_lower = (40, 40, 40)
    hsv_higher = (70, 255, 255)
    green_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)

    return float(np.mean(green_mask))




def get_std_of_green_pixels(image):
    
    '''Return standard deviation of all green pixel values per image'''
    
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get the green mask
    hsv_lower = (40, 40, 40)
    hsv_higher = (70, 255, 255)
    green_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)

    return float(np.std(green_mask))

 

def add_mean_green_pixels(df):
    
    '''Add mean of green pixel values to a given dataframe'''
    
    green = []
    for _, row in df.iterrows():
        img_id = row.image_name
        image = cv2.imread(str(data_dir / img_id))
        green.append(get_mean_of_green_pixels(image))

    green_df = pd.DataFrame(green)
    green_df.columns = ['green_pixel_mean']
    df = pd.concat([df, green_df], ignore_index=True, axis=1)
    df.columns = ['image_name', 'green_pixel_mean']

    return df




def add_std_green_pixels(df):
    
    '''Add standard deviation of yellow pixels per image to a given dataframe'''
    
    green = []
    for _, row in df.iterrows():
        img_id = row.image_name
        image = cv2.imread(str(data_dir / img_id))
        green.append(get_std_of_green_pixels(image))

    green_df = pd.DataFrame(green)
    green_df.columns = ['green_pixel_std']
    df = pd.concat([df, green_df], ignore_index=True, axis=1)
    df.columns = ['image_name', 'green_pixel_std']

    return df




def get_std_brightness(image):
    
    '''Return standard deviation of brightness of all pixels in an image'''
    
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get average brightness
    return np.array(gray).std()

 

def add_std_brightness_column(df):
    
    '''Add standard deviation of brightness of each image to a given dataframe'''
    
    brightness_std = []
    for _, row in df.iterrows():
        img_id = row.image_name
        image = cv2.imread(str(data_dir / img_id))
        brightness_std.append(get_std_brightness(image))

    brightness_std_df = pd.DataFrame(brightness_std)
    brightness_std_df.columns = ['brightness_std']
    df = pd.concat([df, brightness_std_df], ignore_index=True, axis=1)
    df.columns = ['image_name', 'brightness_std']

    return df

 

    
def get_mean_of_BGR_channel(image, channel_num):
    
    '''Return mean intensity of a specified color channel and image'''
    
    channel_values = image[:,:,int(channel_num)]
    return float(np.mean(channel_values))




def add_mean_channel_column(df, channel_num, column_name):

    '''Add column with mean intensity of a specified color channel for each image in a dataframe'''
    
    channel = []
    for _, row in df.iterrows():
        img_id = row.image_name
        image = cv2.imread(str(data_dir / img_id))
        channel.append(get_mean_of_BGR_channel(image, channel_num))

    channel_df = pd.DataFrame(channel)
    channel_df.columns = [str(column_name)]
    df = pd.concat([df, channel_df], ignore_index=True, axis=1)
    df.columns = ['image_name', str(column_name)]

    return df

 

    
def get_std_of_BGR_channel(image, channel_num):
    
    '''Return std of the intensity of a specified color channel and image'''
    
    channel_values = image[:,:,int(channel_num)]
    return float(np.std(channel_values))




def add_std_channel_column(df, channel_num, column_name):
    
    '''Add column with std of the intensity of a specified color channel for each image in a dataframe'''
    
    channel = []
    for _, row in df.iterrows():
        img_id = row.image_name
        image = cv2.imread(str(data_dir / img_id))
        channel.append(get_std_of_BGR_channel(image, channel_num))

    channel_df = pd.DataFrame(channel)
    channel_df.columns = [str(column_name)]
    df = pd.concat([df, channel_df], ignore_index=True, axis=1)
    df.columns = ['image_name', str(column_name)]

    return df