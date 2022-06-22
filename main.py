import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

import Network
from Dataset import train_ds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 로더 생성하기
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

# 손실함수
loss_func_gan = nn.BCELoss()
loss_func_pix = nn.L1Loss()

# 최적화 파라미터

lr = 2e-4
beta1 = 0.5
beta2 = 0.999

# loss_func_pix 가중치
lambda_pixel = 100

# patch 수
patch = (1, 256 // 2 ** 4, 256 // 2 ** 4)
# print("patch: ", patch) # patch.size():  (1, 16, 16)


def train(model_dis, model_gen):
    # 학습
    model_gen.train()
    model_dis.train()

    num_epochs = 100
    start_time = time.time()

    loss_hist = {'gen': [],
                 'dis': []}

    for epoch in range(num_epochs):
        for a, b in train_dl:
            # image a: input(noise) , image b: ground truth
            # print("a.size(): ", a.size()) # torch.Size([32, 3, 256, 256])
            # print("b.size(): ", b.size()) # torch.Size([32, 3, 256, 256])
            ba_si = a.size(0) # 32

            # real image
            real_a = a.to(device) # input image
            real_b = b.to(device) # ground truth

            # patch label
            real_patch_label = torch.ones(ba_si, *patch, requires_grad=False).to(device) # torch.Size([32, 1, 16, 16])
            fake_patch_label = torch.zeros(ba_si, *patch, requires_grad=False).to(device) # torch.Size([32, 1, 16, 16])

            # generator
            model_gen.zero_grad()

            fake_b = model_gen(real_a)  # 가짜 이미지 생성 # torch.Size([32, 3, 256, 256])
            out_dis = model_dis(fake_b, real_b)  # 가짜 이미지 식별 # torch.Size([32, 1, 16, 16])
            # 가짜 image에 대한 condition으로 ground truth를 활용하는 꼴

            gen_loss = loss_func_gan(out_dis, real_patch_label) # discriminator를 속이기 위해 real label로 학습
            pixel_loss = loss_func_pix(fake_b, real_b) # 가짜 이미지와 ground truth를 비교

            g_loss = gen_loss + lambda_pixel * pixel_loss
            g_loss.backward()
            opt_gen.step()

            # discriminator
            model_dis.zero_grad()

            out_dis = model_dis(real_b, real_a)  # 진짜 이미지 식별, ground truth와 input 비교, ground truth가 condition
            real_loss = loss_func_gan(out_dis, real_patch_label)

            out_dis = model_dis(fake_b.detach(), real_a)  # 가짜 이미지 식별
            fake_loss = loss_func_gan(out_dis, fake_patch_label)

            d_loss = (real_loss + fake_loss) / 2.
            d_loss.backward()
            opt_dis.step()

            loss_hist['gen'].append(g_loss.item())
            loss_hist['dis'].append(d_loss.item())
        print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' % (
            epoch, g_loss.item(), d_loss.item(), (time.time() - start_time) / 60))
    return loss_hist


def eval():
    # evaluation model
    model_gen.eval()

    # 가짜 이미지 생성
    with torch.no_grad():
        for a, b in train_dl:
            fake_imgs = model_gen(a.to(device)).detach().cpu()
            real_imgs = b
            break
    return fake_imgs, real_imgs


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    """
    # check
    x = torch.randn(16, 3, 256, 256, device=device)
    model = Network.UNetDown(3, 64).to(device)
    down_out = model(x)
    print(down_out.shape) # torch.Size([16, 64, 128, 128])

    # check
    x = torch.randn(16, 128, 64, 64, device=device)
    model = Network.UNetUp(128, 64).to(device)
    out = model(x, down_out)
    print(out.shape) # torch.Size([16, 128, 128, 128])

    # check
    x = torch.randn(16, 3, 256, 256, device=device)
    model = Network.GeneratorUNet().to(device)
    out = model(x)
    print(out.shape) # torch.Size([16, 3, 256, 256])

    # check
    x = torch.randn(16, 64, 128, 128, device=device)
    model = Network.Dis_block(64, 128).to(device)
    out = model(x)
    print(out.shape) # torch.Size([16, 128, 64, 64])

    # check
    x = torch.randn(16, 3, 256, 256, device=device)
    model = Network.Discriminator().to(device)
    out = model(x, x)
    print(out.shape) # torch.Size([16, 1, 16, 16])
    """
    model_gen = Network.GeneratorUNet().to(device)
    model_dis = Network.Discriminator().to(device)

    # 가중치 초기화 적용
    model_gen.apply(Network.initialize_weights)
    model_dis.apply(Network.initialize_weights)

    opt_dis = optim.Adam(model_dis.parameters(), lr=lr, betas=(beta1, beta2))
    opt_gen = optim.Adam(model_gen.parameters(), lr=lr, betas=(beta1, beta2))

    loss_hist = train(model_dis, model_gen)

    # loss history
    plt.figure(figsize=(10, 5))
    plt.title('Loss Progress')
    plt.plot(loss_hist['gen'], label='Gen. Loss')
    plt.plot(loss_hist['dis'], label='Dis. Loss')
    plt.xlabel('batch count')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 가중치 저장
    path2models = './models/'
    os.makedirs(path2models, exist_ok=True)
    path2weights_gen = os.path.join(path2models, 'weights_gen.pt')
    path2weights_dis = os.path.join(path2models, 'weights_dis.pt')

    torch.save(model_gen.state_dict(), path2weights_gen)
    torch.save(model_dis.state_dict(), path2weights_dis)

    # 가중치 불러오기
    weights = torch.load(path2weights_gen)
    model_gen.load_state_dict(weights)

    fake_imgs, real_imgs = eval()

    # 가짜 이미지 시각화
    plt.figure(figsize=(10, 10))

    for ii in range(0, 16, 2):
        plt.subplot(4, 4, ii + 1)
        plt.imshow(to_pil_image(0.5 * real_imgs[ii] + 0.5))
        plt.axis('off')
        plt.subplot(4, 4, ii + 2)
        plt.imshow(to_pil_image(0.5 * fake_imgs[ii] + 0.5))
        plt.axis('off')
    plt.show()
