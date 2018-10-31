# link num is 45148, feature_dim is 24(coarse), one-hot bag of words

# link num is 45148, feature_dim is 151(fine), one-hot bag of words

# link num is 45148, feature_dim is 68(link_info)
### extract link info feature of each link, 44 cols ###

>     mapid Char(8)
>     id Char(13)
>     kind_num Char(2) # nominal, one-hot
>     kind Char(30) # nominal, one-hot
>     width Char(3) # numeric, float
>     direction Char(1) # nominal, one-hot
>     toll Char(1) # nominal, one-hot
>     const_st Char(1)
>     undconcrid Char(13)
>     snodeid Char(13)
>     enodeid Char(13)
>     pathclass Char(2) # nominal, one-hot
>     length Char(8) # numeric, float
>     detailcity Char(1)
>     through Char(1)
>     unthrucrid Char(13)
>     ownership Char(1)
>     road_cond Char(1)
>     special Char(1)
>     admincodel Char(6)
>     admincoder Char(6)
>     uflag Char(1)
>     onewaycrid Char(13)
>     accesscrid Char(13)
>     speedclass Char(1) # numeric, float
>     lanenums2e Char(2) 
>     lanenume2s Char(2) 
>     lanenum Char(1) # numeric, float
>     vehcl_type Char(32)
>     elevated Char(1)
>     structure Char(1)
>     usefeecrid Char(13)
>     usefeetype Char(1)
>     spdlmts2e Char(4) # nominal, one-hot
>     spdlmte2s Char(4) # nominal, one-hot
>     spdsrcs2e Char(1)
>     spdsrce2s Char(1)
>     spdms2e Char(1)
>     spdme2s Char(1)
>     dc_type Char(1)
>     verify_flag Char(4)
>     walk_form Char(256)
>     pre_launch Char(256)
>     status Char(4)

# feature dim is 6(time_feature)
### extract time feature ###

>     period: 1 Apr, 2017 - 31 May, 2017
>     workday, holiday(weekend or festival): one-hot, 3 dim
>     hour: float, 1 dim
>     min: float, 1 dim
>     peak hour(7:00-10:00, 17:00-20:00): float, 1 dim
>     time_feature_dim = 6
 
# input_dim = 24 + 151 + 68 + 6 = 249

## 前三种feature的pkl文件格式如下：

>      (link_list_coarse, poi_type_feature_coarse) = cPickle.load(open(input_poi_type_feature_coarse_file, "rb"))
>      (link_list_fine, poi_type_feature_fine) = cPickle.load(open(input_poi_type_feature_fine_file, "rb"))
>      (link_list, link_info_feature) = cPickle.load(open(input_event_top5_link_info_feature_beijing_file, "rb"))
其中
> link_list
为一个list，里面存储45158个link_id，feature的格式为np的array，初始化的时候如下：

>     feature = np.zeros((link_num, feature_dim), dtype=np.float)

## time的pkl文件格式如下：
>     time_feature = cPickle.load(open(input_time_feature_file))

其中time_feature是np的array，初始化的时候如下：

>     TOTAL_TIME = 61 * 24 * 4  # 61 days * 24 hours * 4 mins(15 min interval)
>     TIME_FEATURE_DIM = 6
>     time_feature = np.zeros((TOTAL_TIME, TIME_FEATURE_DIM), dtype=np.float)