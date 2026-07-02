import argparse
try:
    from itertools import izip as zip
except ImportError:
    pass
import sys
from random import shuffle
import os


def resolve_data_root(data_root):
    if os.path.isdir(data_root):
        return data_root
    parent_candidate = os.path.join('..', data_root)
    if os.path.isdir(parent_candidate):
        return parent_candidate
    return data_root


def resolve_data_subdir(data_root, explicit_path, preferred_name, fallback_name):
    if explicit_path:
        return explicit_path
    preferred_path = os.path.join(data_root, preferred_name)
    if os.path.isdir(preferred_path):
        return preferred_path
    fallback_path = os.path.join(data_root, fallback_name)
    return fallback_path


def print_dataset_debug(name, dataset):
    print(name + ' size:', len(dataset))
    for index, path in enumerate(dataset.imgs[:5]):
        image_name = os.path.basename(path)
        print(name + ' sample ' + str(index) + ':', image_name, 'label=' + dataset.label_all[image_name])


def load_training_dependencies():
    global torch, transforms, ImageFolder, DataLoader, vutils, trainer
    try:
        import torch
        from torchvision import transforms
        from data import ImageFolder
        from torch.utils.data import DataLoader
        import torchvision.utils as vutils
        from trainer_exchange import trainer
    except ImportError as exc:
        print('CAML training dependency import failed:', exc)
        print('Install PyTorch and torchvision in this Python environment before training.')
        sys.exit(1)


parser = argparse.ArgumentParser()
parser.add_argument('--cuda',type=str,default='True',help='Use gpu or not')
parser.add_argument('--data_root',type=str,default='data/Brain_Tumor2',help='Root folder containing CAML train/test image folders and label txt files')
parser.add_argument('--output_dir',type=str,default='results',help='Root output folder for models and generated training previews')
parser.add_argument('--smoke_test',type=str,default='False',help='Use Brain_smoke defaults and short training settings')
parser.add_argument('--trainA_img_path','--trainA_path',dest='trainA_img_path',type=str,default=None)  ###training images with class A are placed into this folder
parser.add_argument('--trainB_img_path','--trainB_path',dest='trainB_img_path',type=str,default=None)  ###training images with class B are placed into this folder
parser.add_argument('--testA_img_path','--testA_path',dest='testA_img_path',type=str,default=None) ###testing images with class A are placed into this folder
parser.add_argument('--testB_img_path','--testB_path',dest='testB_img_path',type=str,default=None) ###testing images with class B are placed into this folder
parser.add_argument('--train_label_file','--label_train',dest='train_label_file',type=str,default=None)  #### name and label of each training image are presented in each line in this file, like the first line:"brain_image_name1 1", where 1 refers to the abnormal class.
parser.add_argument('--test_label_file','--label_test',dest='test_label_file',type=str,default=None)
parser.add_argument('--style_dim',type=int,default=8) ### the dimension setting of the class-associated codes
parser.add_argument('--train_is',type=str,default='True')

if parser.parse_known_args()[0].train_is=='True':
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--model_save_iter',type=int,default=2000)
    parser.add_argument('--model_save_path',type=str,default=None)
    parser.add_argument('--display_size',type=int,default=16)
    parser.add_argument('--image_save_iter',type=int,default=2000)
    parser.add_argument('--image_save_path',type=str,default=None,help='display of generative cases')
    parser.add_argument('--train_max_iter','--max_iter',dest='train_max_iter',type=int,default=500000)
    parser.add_argument('--learning_rate',type=float,default=0.0001)
    parser.add_argument('--beta1',type=float,default=0.5)
    parser.add_argument('--beta2',type=float,default=0.999)
    parser.add_argument('--weight_decay',type=float,default=0.0001)
    parser.add_argument('--rec_x_w',type=float,default=10)
    parser.add_argument('--rec_c_w',type=float,default=1)
    parser.add_argument('--rec_s_w',type=float,default=1)
    parser.add_argument('--cyc_w',type=float,default=10)
    parser.add_argument('--gen_dis_w',type=float,default=1)
    parser.add_argument('--gen_cla_w',type=float,default=1)
    parser.add_argument('--dis_dis_w',type=float,default=1)
    parser.add_argument('--dis_cla_w',type=float,default=2)


opts = parser.parse_args()

if opts.smoke_test == 'True':
    if opts.data_root == 'data/Brain_Tumor2':
        opts.data_root = 'data/Brain_smoke'
    if opts.train_is == 'True':
        opts.batch_size = 2
        opts.train_max_iter = 5
        opts.model_save_iter = 1
        opts.image_save_iter = 1
        opts.num_workers = 0
        opts.display_size = 2

opts.data_root = resolve_data_root(opts.data_root)
opts.trainA_img_path = resolve_data_subdir(opts.data_root, opts.trainA_img_path, 'trainA_img', 'trainA')
opts.trainB_img_path = resolve_data_subdir(opts.data_root, opts.trainB_img_path, 'trainB_img', 'trainB')
opts.testA_img_path = resolve_data_subdir(opts.data_root, opts.testA_img_path, 'testA_img', 'testA')
opts.testB_img_path = resolve_data_subdir(opts.data_root, opts.testB_img_path, 'testB_img', 'testB')
opts.train_label_file = opts.train_label_file or os.path.join(opts.data_root, 'trainAB_img-name_label.txt')
opts.test_label_file = opts.test_label_file or os.path.join(opts.data_root, 'testAB_img-name_label.txt')
if opts.train_is == 'True':
    opts.model_save_path = opts.model_save_path or os.path.join(opts.output_dir, 'models')
    opts.image_save_path = opts.image_save_path or os.path.join(opts.output_dir, 'images_display')

print('Data root:', opts.data_root)
print('TrainA:', opts.trainA_img_path)
print('TrainB:', opts.trainB_img_path)
print('TestA:', opts.testA_img_path)
print('TestB:', opts.testB_img_path)
print('Train labels:', opts.train_label_file)
print('Test labels:', opts.test_label_file)

load_training_dependencies()

if opts.cuda=='True' and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

optim_para={'lr':opts.learning_rate,'lr_step_size':max(1, int(opts.train_max_iter/10)),'beta1':opts.beta1,'beta2':opts.beta2,'weight_decay':opts.weight_decay}
gen_loss_weight_para={'rec_x_w':opts.rec_x_w,'rec_c_w':opts.rec_c_w,'rec_s_w':opts.rec_s_w,'cyc_w':opts.cyc_w,'gen_dis_w':opts.gen_dis_w,'gen_cla_w':opts.gen_cla_w}
dis_loss_weight_para={'dis_dis_w':opts.dis_dis_w,'dis_cla_w':opts.dis_cla_w}

train_is=opts.train_is

trainer = trainer(device=device,style_dim=opts.style_dim,optim_para=optim_para,gen_loss_weight_para=gen_loss_weight_para,dis_loss_weight_para=dis_loss_weight_para,train_is=train_is)
trainer.to(device)

# image path
trainA_img_path=opts.trainA_img_path
trainB_img_path=opts.trainB_img_path
testA_img_path=opts.testA_img_path
testB_img_path=opts.testB_img_path

# label path
train_label_file=opts.train_label_file
test_label_file=opts.test_label_file


# data loader
transform1=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform2=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset_trainA=ImageFolder(trainA_img_path, train_label_file,transform=transform1)
dataset_trainB=ImageFolder(trainB_img_path, train_label_file,transform=transform1)
dataset_testA=ImageFolder(testA_img_path, test_label_file,transform=transform2)
dataset_testB=ImageFolder(testB_img_path, test_label_file,transform=transform2)
print_dataset_debug('TrainA', dataset_trainA)
print_dataset_debug('TrainB', dataset_trainB)
print_dataset_debug('TestA', dataset_testA)
print_dataset_debug('TestB', dataset_testB)
batch_size=opts.batch_size
num_workers=opts.num_workers
train_loader_a=DataLoader(dataset=dataset_trainA, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
train_loader_b=DataLoader(dataset=dataset_trainB, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
test_loader_a=DataLoader(dataset=dataset_testA, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
test_loader_b =DataLoader(dataset=dataset_testB, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)


## display set
display_size=opts.display_size
def display_a2b_b2a_images(image_outputs, display_image_num, image_save_path, image_iter):
    n = len(image_outputs)

    ###a2b_display
    a2b_image_ouputs=image_outputs[0:n//2]
    a2b_image_ouputs = [images.expand(-1, 3, -1, -1) for images in a2b_image_ouputs]  # expand gray-scale images to 3 channels
    a2b_image_tensor = torch.cat([images[:display_image_num] for images in a2b_image_ouputs], 0)
    a2b_image_grid = vutils.make_grid(a2b_image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(a2b_image_grid, '%s/gen_a2b_%s.jpg' % (image_save_path, image_iter), nrow=1)

    ###b2a_display
    b2a_image_ouputs = image_outputs[n//2:n]
    b2a_image_ouputs = [images.expand(-1, 3, -1, -1) for images in
                        b2a_image_ouputs]  # expand gray-scale images to 3 channels
    b2a_image_tensor = torch.cat([images[:display_image_num] for images in b2a_image_ouputs], 0)
    b2a_image_grid = vutils.make_grid(b2a_image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(b2a_image_grid, '%s/gen_b2a_%s.jpg' % (image_save_path, image_iter), nrow=1)


# training samples display
ta_list=[]
for i in range(len(train_loader_a.dataset)):
    ta_list.append(i)
tb_list=[]
for i in range(len(train_loader_b.dataset)):
    tb_list.append(i)
shuffle(ta_list)
train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in ta_list[:display_size]]).to(device)
shuffle(tb_list)
train_display_images_b1 = torch.stack([train_loader_b.dataset[i][0] for i in tb_list[:display_size]]).to(device)
shuffle(tb_list)
train_display_images_b2 = torch.stack([train_loader_b.dataset[i][0] for i in tb_list[:display_size]]).to(device)
shuffle(tb_list)
train_display_images_b3 = torch.stack([train_loader_b.dataset[i][0] for i in tb_list[:display_size]]).to(device)
shuffle(tb_list)
train_display_images_b4 = torch.stack([train_loader_b.dataset[i][0] for i in tb_list[:display_size]]).to(device)
shuffle(tb_list)
train_display_images_b5 = torch.stack([train_loader_b.dataset[i][0] for i in tb_list[:display_size]]).to(device)
# testing samples display
ea_list=[]
for i in range(len(test_loader_a.dataset)):
    ea_list.append(i)
eb_list=[]
for i in range(len(test_loader_b.dataset)):
    eb_list.append(i)
shuffle(ea_list)
test_display_images_a = torch.stack([test_loader_a.dataset[i][0] for i in ea_list[:display_size]]).to(device)
shuffle(eb_list)
test_display_images_b1 = torch.stack([test_loader_b.dataset[i][0] for i in eb_list[:display_size]]).to(device)
shuffle(eb_list)
test_display_images_b2 = torch.stack([test_loader_b.dataset[i][0] for i in eb_list[:display_size]]).to(device)
shuffle(eb_list)
test_display_images_b3 = torch.stack([test_loader_b.dataset[i][0] for i in eb_list[:display_size]]).to(device)
shuffle(eb_list)
test_display_images_b4 = torch.stack([test_loader_b.dataset[i][0] for i in eb_list[:display_size]]).to(device)
shuffle(eb_list)
test_display_images_b5 = torch.stack([test_loader_b.dataset[i][0] for i in eb_list[:display_size]]).to(device)



if __name__=='__main__':
    # Start training
    iterations = 0
    image_save_iter=opts.image_save_iter
    model_save_iter=opts.model_save_iter
    image_save_path=opts.image_save_path
    model_save_path = opts.model_save_path
    max_iter=opts.train_max_iter
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    while True:
        for it, (images_label_a, images_label_b) in enumerate(zip(train_loader_a, train_loader_b)):

            ##image a and b with label
            images_a=images_label_a[0]
            a_label=images_label_a[1]
            images_b=images_label_b[0]
            b_label=images_label_b[1]
            images_a, images_b = images_a.to(device).detach(), images_b.to(device).detach()
            a_label,b_label=a_label.to(device),b_label.to(device)

            # discriminator update
            trainer.dis_update(images_a, images_b,dis_a_real_label=a_label,dis_b_real_label=b_label)
            # encoder and decoder update
            trainer.gen_update(images_a, images_b,dis_a_real_label=a_label,dis_b_real_label=b_label)
            #learning rate update
            trainer.update_learning_rate()

            # save display images
            if (iterations + 1) % image_save_iter == 0:
                with torch.no_grad():
                    test_image_outputs1 = trainer.display(test_display_images_a, test_display_images_b1)
                    train_image_outputs1 = trainer.display(train_display_images_a, train_display_images_b1)
                    test_image_outputs2 = trainer.display(test_display_images_a, test_display_images_b2)
                    train_image_outputs2 = trainer.display(train_display_images_a, train_display_images_b2)
                    test_image_outputs3 = trainer.display(test_display_images_a, test_display_images_b3)
                    train_image_outputs3 = trainer.display(train_display_images_a, train_display_images_b3)
                    test_image_outputs4 = trainer.display(test_display_images_a, test_display_images_b4)
                    train_image_outputs4 = trainer.display(train_display_images_a, train_display_images_b4)
                    test_image_outputs5 = trainer.display(test_display_images_a, test_display_images_b5)
                    train_image_outputs5 = trainer.display(train_display_images_a, train_display_images_b5)

                display_a2b_b2a_images(test_image_outputs1, display_size, image_save_path, 'test1_%08d' % (iterations + 1))
                display_a2b_b2a_images(train_image_outputs1, display_size, image_save_path, 'train1_%08d' % (iterations + 1))
                display_a2b_b2a_images(test_image_outputs2, display_size, image_save_path, 'test2_%08d' % (iterations + 1))
                display_a2b_b2a_images(train_image_outputs2, display_size, image_save_path, 'train2_%08d' % (iterations + 1))
                display_a2b_b2a_images(test_image_outputs3, display_size, image_save_path, 'test3_%08d' % (iterations + 1))
                display_a2b_b2a_images(train_image_outputs3, display_size, image_save_path, 'train3_%08d' % (iterations + 1))
                display_a2b_b2a_images(test_image_outputs4, display_size, image_save_path, 'test4_%08d' % (iterations + 1))
                display_a2b_b2a_images(train_image_outputs4, display_size, image_save_path, 'train4_%08d' % (iterations + 1))
                display_a2b_b2a_images(test_image_outputs5, display_size, image_save_path, 'test5_%08d' % (iterations + 1))
                display_a2b_b2a_images(train_image_outputs5, display_size, image_save_path, 'train5_%08d' % (iterations + 1))

            # Save model
            if (iterations + 1) % model_save_iter == 0:
                trainer.save(model_save_path, iterations)



            iterations += 1

            print('Finished training: '+str(iterations)+'/'+str(max_iter))
            if iterations >= max_iter:
                print('Training finish')
                sys.exit(0)
