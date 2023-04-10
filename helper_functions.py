import matplotlib.pyplot as plt
import imageio

def labels_gif(X, labels_array, title, images_dir, filename):
    '''For 2D data, creates a gif from all the labelings visited by the algorithm'''

    time = range(0,len(labels_array))

    def create_frame(t):
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels_array[t], s=40, cmap='Accent')
        plt.title(f'{title}, Iter: {t}',fontsize=14)
        plt.savefig(f'{images_dir}/gif_frames/img_{t}.png', transparent = False, facecolor = 'white')
        plt.close()

    for t in time:
        create_frame(t)

    frames = []
    
    for t in time:
        image = imageio.imread(f'{images_dir}/gif_frames/img_{t}.png')
        frames.append(image)

    imageio.mimsave(f'{images_dir}/{filename}.gif', frames, fps = 5)
