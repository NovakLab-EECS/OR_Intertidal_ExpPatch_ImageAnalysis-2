import type { FC } from "react";
import { useCallback } from "react";
import { Accept, useDropzone } from 'react-dropzone'

interface IDropzoneProps {
  callback: (acceptedFiles: any) => void;
  accept?: Accept;
}

const Dropzone: FC<IDropzoneProps> = ({ callback, accept }) => {

  const onDrop = useCallback(callback, [])

  const { getRootProps,
    getInputProps,
    isDragActive } = useDropzone({
      onDrop, accept
    })


  return (
    <div {...getRootProps()}>
      <input {...getInputProps()} />
      {
        isDragActive ?
          <p>Drop the files here ...</p> :
          <p>Drag 'n' drop some files here, or click to select files</p>
      }
    </div>
  )
}

export default Dropzone