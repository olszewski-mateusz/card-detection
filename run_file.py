import numpy as np
import utils as ut
import matplotlib.pyplot as plt
from skimage import io

upper_symbols = [(io.imread('templates/0.bmp')[:,:,0], '10'),
                 (io.imread('templates/9.bmp')[:,:,0], '9'),
                 (io.imread('templates/J.bmp')[:,:,0], 'J'),
                 (io.imread('templates/Q.bmp')[:,:,0], 'Q'),
                 (io.imread('templates/K.bmp')[:,:,0], 'K'),
                 (io.imread('templates/A.bmp')[:,:,0], 'A')]

lower_symbols = [(io.imread('templates/Kier.bmp')[:,:,0], 'Kier'),
                 (io.imread('templates/Karo.bmp')[:,:,0], 'Karo'),
                 (io.imread('templates/Pik.bmp')[:,:,0], 'Pik'),
                 (io.imread('templates/Trefl.bmp')[:,:,0], 'Trefl')]
src = "input/img3.jpg"
image = io.imread(src)

fig, ax = plt.subplots(figsize=(30, 30))
fig.tight_layout(pad=0.0)
preparedImage = ut.prepareImage(image)
card_contours = ut.getCardsContours(preparedImage)

for card_contour in card_contours:
    try: corners1,corners2 = ut.findCorners(card_contour)
    except ut.Error: continue;
        
    upper_ranking = []
    lower_ranking = []
    
    for corners, crn_str in [(corners1, 'c1'), (corners2, 'c2')]:
        for q in [0.20,0.21,0.19,0.22,0.18]:
            warpedCard = ut.warpCard(image, corners)
            cuttedCard = ut.cutCard(warpedCard)
            binaryCard = ut.thresholdCard(cuttedCard, q=q)
            preparedCard = ut.prepareCard(binaryCard)
            symbols = ut.getSymbolsContours(preparedCard)
                
            upper_sym = None
            lower_sym = None
            upper_max_score = 0
            lower_max_score = 0
            
            for symbol_contour in symbols:
                y = np.mean(symbol_contour[:,0])
                x = np.mean(symbol_contour[:,1])
                
                cuttedSymbol = ut.cutSymbol(symbol_contour, binaryCard)   
                if x > 20 and x < 45 and y > 35 and y < 70:
                    for template, text in upper_symbols:
                        score_matrix = ut.compareSymbolWithTemplate(cuttedSymbol, template)
                        score = np.mean(score_matrix)
                        if(score > upper_max_score):
                            upper_max_score = score
                            upper_sym = text
                            
                if x > 20 and x < 45 and y > 90 and y < 125:
                    for template, text in lower_symbols:
                        score_matrix = ut.compareSymbolWithTemplate(cuttedSymbol, template)
                        score = np.mean(score_matrix)
                        if(score > lower_max_score):
                            lower_max_score = score
                            lower_sym = text
            upper_ranking.append((upper_max_score, upper_sym, crn_str, q))
            lower_ranking.append((lower_max_score, lower_sym, crn_str, q))
            if ut.is_correct_match(upper_max_score, lower_max_score):
                break
        else:
            continue
        break
    upper_ranking.sort(reverse=True)
    lower_ranking.sort(reverse=True)
    
    upper_max_score, upper_sym = upper_ranking[0][0:2]
    lower_max_score, lower_sym = lower_ranking[0][0:2]
        
    if not ut.is_correct_match(upper_max_score, lower_max_score):
        continue
                        
    y = np.mean(card_contour[:,0])
    x = np.mean(card_contour[:,1])

    text = upper_sym + ': ' + str(round(upper_max_score,2)) + '\n' + lower_sym + ' ' + str(round(lower_max_score,2))
    ax.text(x,y, text, fontsize=30, bbox=dict(facecolor='white', alpha=0.7))
    for i in range(4):
        j = (i+1)%4
        ax.plot([corners[i][0], corners[j][0]], 
                [corners[i][1], corners[j][1]], 
                linewidth=7, color='blue')   
ax.imshow(image)
ax.tick_params(top=False, bottom=False, left=False, right=False,
            labelleft=False, labelbottom=False)

plt.savefig('output.png')