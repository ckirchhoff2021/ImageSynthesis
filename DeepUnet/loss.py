import torch

def content_loss(criterion, xs, ys):
    '''
    calculate content loss
    '''
    content = 0.0
    count = len(xs)
    for i in range(count):
        xi, yi = xs[i], ys[i]
        loss = criterion(xi, yi)
        content += loss
    return content


def fft_loss(criterion, xs, ys):
    '''
    calculate fft loss
    '''
    loss = 0.0
    count = len(xs)
    for i in range(count):
        xi, yi = xs[i], ys[i]
        y_fft = torch.rfft(yi, signal_ndim=2, normalized=False, onesided=False)
        x_fft = torch.rfft(xi, signal_ndim=2, normalized=False, onesided=False)
        l = criterion(x_fft, y_fft)
        loss += l
    return loss


def perceptual_loss(criterion, xs, ys):
    '''
    calculate vgg loss
    '''
    loss = 0.0
    count = len(xs)
    for i in range(count):
        xi, yi = xs[i], ys[i]
        l = criterion(xi, yi)
        loss += l
    return loss


def gradient_loss(criterion, xs, ys):
    '''
    calculate vgg loss
    '''
    loss = 0.0
    count = len(xs)
    for i in range(count):
        xi, yi = xs[i], ys[i]
        l = criterion(xi, yi)
        loss += l
    return loss


def adversarial_loss(criterion, probsr, probsf, real, fake):
    '''
    calculate adversarial loss
    '''

    '''
    probsrf = probsr - probsf.mean()
    probsfr = probsf - probsr.mean()
    adv_rf = criterion(probsrf, real)
    adv_fr = criterion(probsfr, fake)
    adv_loss = (adv_rf + adv_fr) / 2
    '''

    adv_real = criterion(probsr, real)
    adv_fake = criterion(probsf, fake)
    adv_loss = (adv_real + adv_fake) / 2

    return adv_loss



def patch_discrimiantor_loss(criterion, discriminator, real_img, fake_img, real_label, fake_label):
    '''
    calculate patch discrimiantor loss
    '''
    _, c, w, h = real_img.size()
    num = w // 16
    loss = 0.0
    for i in range(num):
        for j in range(num):
            row = 16 * i
            col = 16 * j
            real = real_img[:, :, row:row+16, col:col+16]
            fake = fake_img[:, :, row:row+16, col:col+16]
            real_probs = discriminator(real)
            fake_probs = discriminator(fake)
            real_loss = criterion(real_probs, real_label)
            fake_loss = criterion(fake_probs, fake_label)
            loss += (real_loss + fake_loss) * 0.5
    
    loss = loss / (num * num)
    return loss
    


def patch_generator_loss(criterion, discriminator, gen_img, real_label):
    '''
    calculate patch generator loss
    '''
    _, c, w, h = gen_img.size()
    num = w // 16
    loss = 0.0
    for i in range(num):
        for j in range(num):
            row = 16 * i
            col = 16 * j
            patch = gen_img[:, :, row:row+16, col:col+16]
            probs = discriminator(patch)
            gen_loss = criterion(probs, real_label)
            loss += gen_loss
    loss = loss / (num * num)
    return loss



