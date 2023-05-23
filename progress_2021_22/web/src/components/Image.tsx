import { FC } from 'react';
import { LazyLoadImage } from 'react-lazy-load-image-component';

interface ILazyLoaderProps {
  alt: string;
  src: string;
  [key: string]: any;
}

const LazyLoader: FC<ILazyLoaderProps> = ({
  alt,
  src,
  ...props
}) => {
  return (
    <LazyLoadImage alt={alt} src={src}

      effect="blur"
      {...props} />
  )
}

export default LazyLoader;