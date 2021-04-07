# chaotic_pokemon
GAN approach pokemon generator


# 과제: 비선형회귀분석

```
using CSV, DataFrames

n_obs = 100

x = 1:n_obs
y = 10rand(n_obs)
z = 10rand(n_obs)

f(x,y,z) = 2sin(x) + y^2 + 10/z

CSV.write("data1.csv",DataFrame(hcat(f.(x,y,z),x,y,z),["f","x","y","z"]))
```

`data1.csv` 파일은 위의 줄리아 코드로 생성된 데이터셋입니다. x,y,z로써 f를 예측하는 인공지능을 구현합시다.

## 조건

1. 어떤 언어와 API를 사용하든 상관 없지만, 딥러닝이어야한다.
2. 기한은 4월 21일. 당일 스터디 이전에 이 원격 저장소에 올라와야한다.
3. 주피터 등으로 온라인에서 실행할 수 있는 코드여야한다.
