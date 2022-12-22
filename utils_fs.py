import os
import shutil
from typing import Optional, Union


def createFolder(folder_path: str) -> None:
    if not pathExists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


def getBasename(filepath: str) -> str:
    return os.path.basename(filepath)


def getDirName(filepath: str) -> str:
    return os.path.dirname(filepath)


def pathExists(filepath: str) -> bool:
    return os.path.exists(filepath)


def removeFileExtension(filepath: str) -> str:
    return os.path.splitext(filepath)[0]


def pathJoin(base: str, *others) -> str:
    return os.path.join(base, *others)


def deleteFile(filepath: str) -> bool:
    if pathExists(filepath):
        os.remove(filepath)
        return True
    return False


def copyFile(src_filepath: str, dst_filepath: str) -> bool:
    if pathExists(src_filepath) and not pathExists(dst_filepath):
        shutil.copy(src_filepath, dst_filepath)
        return True
    return False


def moveFile(src_filepath: str, dst_filepath: str) -> bool:
    if pathExists(src_filepath) and not pathExists(dst_filepath):
        os.replace(src_filepath, dst_filepath)
        return True
    return False


def deleteFolder(folder_path: str) -> bool:
    if pathExists(folder_path):
        shutil.rmtree(folder_path)
        return True
    return False


def isFile(filepath: str) -> bool:
    return os.path.isfile(filepath) or os.path.islink(filepath)


def isFolder(filepath: str) -> bool:
    return os.path.isdir(filepath)


def deletePath(filepath: str) -> bool:
    if isFolder(filepath):
        return deleteFolder(filepath)
    elif isFile(filepath):
        return deleteFile(filepath)
    return False


def deleteFolderContents(folder_path: str, ignore: Optional[Union[list, set]] = None) -> bool:
    if pathExists(folder_path):
        ok = True
        for filename in os.listdir(folder_path):
            if filename not in ignore:
                file_path = pathJoin(folder_path, filename)
                if not deletePath(file_path):
                    ok = False
        return ok
    return False
