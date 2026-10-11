# YOLO-World 모델 (Git LFS)

이 폴더도 네 Python 파일과 함께 GitHub에 업로드합니다.

- `yolov8x-worldv2.pt`: 원본에서 복사한 YOLO-World 모델
- 크기: 146,355,704바이트(약 139.6MiB)
- GitHub의 일반 파일 제한을 넘으므로 루트 `.gitattributes`에 Git LFS로 등록되어 있습니다.

## 업로드

Git과 Git LFS를 설치한 뒤 `Visoin_AI_Github_Project` 저장소 루트에서 실행하세요.

```powershell
git init
git lfs install --local
git add .
git lfs ls-files
```

목록에 모델이 표시되는지 확인한 뒤 `.gitattributes`, 코드, 모델을 함께 commit/push합니다.
브라우저 파일 업로드 대신 Git LFS를 사용하세요.

## 다운로드와 실행

Git LFS가 설치된 환경에서 저장소를 clone하고, 저장소 루트에서 실행하세요.

```powershell
git lfs install --local
git lfs pull
```

YOLO-World 코드가 이 폴더의 실제 모델을 읽습니다. 모델 파일이 없거나 LFS 포인터만 내려받았으면
다운로드 안내를 표시합니다. 별도의 모델을 쓰려면 `--model` 또는 `YOLO_WORLD_MODEL`로 지정하세요.
GitHub ZIP에 실제 모델이 포함되지 않았다면 Git LFS를 사용하는 `git clone`으로 받아야 합니다.

CLIP 텍스트 인코더와 DETR의 `facebook/detr-resnet-50`은 이 파일과 별개로 최초 실행 시 다운로드됩니다.

[GitHub Git LFS 설정 안내](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage)
