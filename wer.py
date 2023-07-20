import jieba

def chinese_word_error_rate(reference, hypothesis):
    reference_words = list(jieba.cut(reference))
    hypothesis_words = list(jieba.cut(hypothesis))

    # 动态规划计算编辑距离
    dp = [[0] * (len(hypothesis_words) + 1) for _ in range(len(reference_words) + 1)]

    for i in range(len(reference_words) + 1):
        dp[i][0] = i

    for j in range(len(hypothesis_words) + 1):
        dp[0][j] = j

    for i in range(1, len(reference_words) + 1):
        for j in range(1, len(hypothesis_words) + 1):
            if reference_words[i - 1] == hypothesis_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

    # 计算词错误率
    errors = dp[len(reference_words)][len(hypothesis_words)]
    if len(reference_words) == 0:
        wer = 1
    else:
        wer = errors / len(reference_words)
    if wer>1:
        wer=1
    return wer
