from utils import balance_df, plot_batch
#from cutmix_keras import CutMixImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageFromDFLoader(object):
    
    """Creates ImageFromDFLoader class that will read from a dataframe.
    Inputs:
        1. df with columns:
            -'image_id' - relative paths from data_dir to image files.
            -'label' - one-hot encoded class column.
        2. label_key - series or dictionary that maps df labels to class strings, i.e. label_key['label']
           returns the corresponding 'class' string.
        3. data_dir: path to directory containing image files.
        4. target_size: image size in pixels for square image (default 380).
        5. batch_size: number of images in each generated batch (default 16).
        6. balanced - if True resamples image df for equal class representation (default False).
        7. class_size - size of each class if balanced set to True (default 2000).
        8. cutmix - boolean, if True then get_train_data generates cutmix images (default False)."""
    
    def __init__(self, df, label_key, data_dir, target_size=380, batch_size=16, balanced=False, class_size=2000):
        self.df = df
        self.label_key = label_key
        self.df['class'] = self.df['label'].apply(lambda x: self.label_key[x])
        self.data_dir = data_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.balanced = balanced
        if self.balanced:
            self.class_size=class_size
        else:
            self.class_size=None 
        self.train_gen, self.val_gen, self.eval_gen = self.get_train_data()
        

    def get_train_data(self):
        
        """Returns train and validation image data generators."""
        data_dir = self.data_dir        
        if self.balanced:
            df = balance_df(self.df, label='label', class_size=self.class_size)
        else:
            df = self.df
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=360,
            brightness_range=[0.5, 1.2],
            shear_range=15,
            zoom_range=0.2,
            height_shift_range=0.1,
            width_shift_range=0.1,
            channel_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=0.15,
        )
        
        '''
        if self.cutmix:
            train_generator1 = train_datagen.flow_from_dataframe(
                df, 
                data_dir, 
                target_size=(self.target_size, self.target_size), 
                x_col='image_id', 
                y_col='class', 
                color_mode='rgb', 
                batch_size=self.batch_size, 
                shuffle=True,
                subset='training'
            )

            train_generator2 = train_datagen.flow_from_dataframe(
                df, 
                data_dir, 
                target_size=(self.target_size, self.target_size), 
                x_col='image_id', 
                y_col='class', 
                color_mode='rgb', 
                batch_size=self.batch_size, 
                shuffle=True,
                subset='training'
            )

            train_generator = CutMixImageDataGenerator(
                generator1=train_generator1, 
                generator2=train_generator2, 
                img_size=self.target_size, 
                batch_size=self.batch_size
            )
         '''   
        
        train_generator = train_datagen.flow_from_dataframe(
            df,
            data_dir,
            x_col='image_id',
            y_col='class',
            target_size=(self.target_size, self.target_size),
            batch_size=self.batch_size,
            shuffle=True,
            subset='training'
            )
                
        val_generator = train_datagen.flow_from_dataframe(
            df, 
            data_dir, 
            target_size=(self.target_size, self.target_size), 
            x_col='image_id', 
            y_col='class', 
            color_mode='rgb', 
            batch_size=self.batch_size, 
            shuffle=True,
            subset='validation',
            seed=1
        )

                
        eval_generator = ImageDataGenerator(rescale=1./255, validation_split=0.15).flow_from_dataframe(
            df, 
            data_dir, 
            target_size=(self.target_size, self.target_size), 
            x_col='image_id', 
            y_col='class', 
            color_mode='rgb', 
            batch_size=self.batch_size, 
            shuffle=False,
            subset='validation',
            seed=1
        )
        
        return train_generator, val_generator, eval_generator
    
    def plot_train_batch(self, generator='train'):
        """Plots a batch of images. For generator choose between 'train', 'val', and 'eval'."""
        if generator == 'train':
            gen = self.train_gen
        elif generator =='val':
            gen = self.val_gen
        elif generator == 'eval':
            gen = self.eval_gen
        plot_batch(gen, label_key=self.label_key, cutmix=self.cutmix)