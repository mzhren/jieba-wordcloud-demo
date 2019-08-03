# - * - coding: utf - 8 -*-

'''
Created on 2019年02月21日
@author: http://www.mzh.ren/
'''

# 导入模块（工具包）
# 操作系统，正则表达式工具，系统工具
import os,re,sys
# 编码转换
import codecs
# csv 用于生成scv文件，类似excel 文件
import csv
# 文件名判断工具
import fnmatch
# 科学计算模块
import numpy as np
# 图像处理模块
from PIL import Image
# 结巴分词工具
import jieba
# 导入数据类型模块collections的子类OrderedDict
from collections import OrderedDict
# matplotlib 是画图模块
import matplotlib.pyplot as plt
# SciPy 是基于Numpy构建的一个集成了多种数学算法和方便的函数的Python模块
from scipy.misc import imread
# wordclound是一个基于python对词频进行绘制图片的工具。
from wordcloud import WordCloud, ImageColorGenerator


 
reload(sys)
sys.setdefaultencoding('utf-8')

current_dir = os.path.dirname(__file__)

debug = 0


# 用于生成csv表格的几个常量
header_filed = ['行业']
csv_dict = dict()
words_list = []


def init_csv_col(col_txt_path):
    # 表头
    fields = open(col_txt_path,'r')
    try:
        fields_text = fields.read()
        fields_text = unicode(fields_text,'utf-8')
    finally:
        fields.close();

    fields_text_list = fields_text.split('\n')
    for item in fields_text_list:
        csv_dict[item] = {}


def word_frequency_analysis(path):
    '''
    对某目录下所有的 txt 文本文件进行词频分析
    该目下创建一个新目录 result 存储每个txt文件的词频分析结果
    分析结果包括词频全文分词统计及行业关键词分析统计
    '''
    # 根据 韶关或清远的特定关键字 生成表头
    # init_csv_col('qy_keywords.txt')
    init_csv_col('sg_keywords.txt')

    global debug
    # 用于调试
    if debug:
        print path

    #该目录下所有文件的名字
    files = os.listdir(path) 
    #该目下创建一个新目录 result，用来分析结果
    result_dir = os.path.abspath(os.path.join(path, 'result'))

    csv_all = os.path.abspath(os.path.join(result_dir, 'csv_all.csv'))

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if debug:
        print result_dir
    for filename in files:
        if debug:
            print filename
        if not fnmatch.fnmatch(filename, '*.txt') :
             continue;
        txt_path = os.path.join(path,filename)
        
        txt_content = open(txt_path,'r').read()
        field_name = filename[:-4] + '年'
        header_filed.append(field_name)
        filename_fulltext = filename[:-4] + '_all.txt'
        filename_counter = filename[:-4] + '_tj.csv'
        filename_key = filename[:-4] + '_hy_tj.csv'
        if debug:
            print filename_fulltext
        txt_to_all = os.path.join(os.path.join(path, 'result'),filename_fulltext)
        txt_to_counter = os.path.join(os.path.join(path, 'result'),filename_counter)
        txt_to_key = os.path.join(os.path.join(path, 'result'),filename_key)

        # 做分词以
        text_cutted = jiebaCutText(txt_content)
        text_cleared = clearText(text_cutted)
        text_counted = countWords(text_cleared,txt_to_counter)
        text_exacted = exactSearch(text_cleared,filename_key,field_name)

        # 将分词好的文本放到新生成的文件
        newfile = open(txt_to_all,'w')
        newfile.write(text_cleared.encode('utf-8'))
        newfile.close()
        
    dataToCsv(csv_all)

    # 准备画图用的轮廓图
    # 清远轮廓图
    # mask_img = os.path.join(current_dir,'qy_bg.jpg');
    # 韶关轮廓图
    mask_img = os.path.join(current_dir,'sg_bg.jpg');
    # 开始画图
    drawImage(current_dir,mask_img)

# 将词频分析的统计数据输出为CSV文件
def dataToCsv(csv_all):
    f = open(csv_all,'w')
    f.write(codecs.BOM_UTF8)
    w = csv.DictWriter(f,header_filed)
    w.writeheader()
    for key,val in sorted(csv_dict.items()):
        row = {'行业': key}
        row.update(val)
        w.writerow(row)
    print 'export to csv done'

# 用jieba分词
def jiebaCutText(text):
    seg_list = jieba.cut(text, cut_all=False)
    liststr=" / ".join(seg_list)
    return liststr

# 过滤不需要统计的词（有些词没意义）
def clearText(text):
    mywordlist = []
    stopwords_path = 'stopwords.txt' # 停用词词表
    f_stop = open(stopwords_path)
    try:
        f_stop_text = f_stop.read()
        f_stop_text=unicode(f_stop_text,'utf-8')
    finally:
        f_stop.close( )
    f_stop_seg_list=f_stop_text.split('\n')
    for myword in text.split(' / '):
        if not(myword.strip() in f_stop_seg_list) and len(myword.strip())>1 and contain_zh(myword.strip()):
            mywordlist.append(myword.strip())
    return '/'.join(mywordlist)

# 精确搜索
def exactSearch(clearedText,keySearchFile,subkey):
    f_key_seg_list = csv_dict.keys()
    count_dict = dict()
    for myword in clearedText.split('/'):
        if myword.strip() in f_key_seg_list:
            if myword in count_dict:
                count_dict[myword] += 1
                words_list.append(myword)
                csv_dict[myword][subkey] +=1
            else:
                csv_dict[myword][subkey] =1
                count_dict[myword] = 1 

# 统计词频
def countWords(text,counter_file):
    count_dict = dict()
    for item in text.split('/'):
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1 
    
    d_sorted_by_value = OrderedDict(sorted(count_dict.items(), key=lambda x: x[1]))
    with open(counter_file,'wb') as f:
        f.write(codecs.BOM_UTF8)
        w = csv.writer(f)
        w.writerows(d_sorted_by_value.items())     

# 生成词云图
def drawImage(path,img_bg):

    bg_img = np.array(Image.open(img_bg))


    font_path = os.path.join(current_dir,'fonts/test.otf')
    wc = WordCloud(font_path=font_path,  # 设置字体
               background_color='white',  # 背景颜色
            #    max_words=10000,  # 词云显示的最大词数
               mask=bg_img,  # 设置背景图片
               max_font_size=200,  # 字体最大值
               contour_width=2, contour_color='firebrick',
               width=1000, height=1000, margin=2
               # 设置图片默认的大小,但是如果使用背景图片的话,那么保存的图片大小将会按照其大小保存,margin为词语边缘距离
               )
    text = ' '.join(words_list)
    print text
    wc.generate(text)

    # 保存图片
    wc.to_file(os.path.join(path, 'sg.png'))

    # 显示图片
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.figure()
    plt.show()

    
    

def contain_zh(word):
    '''
    判断传入字符串是否包含中文
    :param word: 待判断字符串
    :return: True:包含中文  False:不包含中文
    '''
    word = word.decode()
    # 正则表达式，用于匹配中文
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    match = zh_pattern.search(word)
    return match


if __name__ == '__main__':
    print u'''
        对某目录下所有的 txt 文本文件进行词频分析
        该目下创建一个新目录 result 存储每个txt文件的词频分析结果
        分析结果包括词频全文分词统计及行业关键词分析统计
    '''
    print(u'请输入你要分析文件所在目录:')
    # mypath = raw_input()
    # 上行代码被注释掉了，用途就是输入词频基础数据所在的文件夹，被注释的原因是下面直接给出了绝对路径
    mypath = u'C:\projects\py\基础数据\韶关（分年度）'
    # mypath = u'C:\projects\py\基础数据\清远（分年度）'
    print (u'正在做词频分析...')
    word_frequency_analysis(mypath)
    raw_input()