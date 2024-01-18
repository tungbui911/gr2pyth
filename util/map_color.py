import math

def map_color(score_list, start_index=1) -> dict:
    list_result = list()
    for word in score_list:
        for right_phoneme, model_predict_phoneme, right_phoneme_score, predict_score in word:
            if right_phoneme == model_predict_phoneme:
                final_score = max(right_phoneme_score, predict_score)**(math.log(0.8, 0.2))
                if final_score >= 0.8:
                    tag = "#28B463"
                else:
                    tag = '#D4AC0D'
            else:
                final_score = right_phoneme_score**(math.log(0.4, 0.05))
                if final_score < 0.4:
                    tag = '#CB4335'
                else:
                    tag = '#D4AC0D'
            list_result.append((right_phoneme, model_predict_phoneme, final_score, tag))
        list_result.append((" "," ", 1, 'black'))

    if len(list_result) > 0:
        list_result.pop()
    return list_result