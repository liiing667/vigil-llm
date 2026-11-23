import math

from loguru import logger

from typing import List, Dict

from vigil.schema import BaseScanner
from vigil.schema import ScanModel
from vigil.schema import ResponseModel
from vigil.schema import ScannerResultModel
from vigil.schema import StatusEmum

from vigil.common import timestamp_str


messages = {
    'scanner:yara': 'Potential prompt injection detected: YARA signature(s)',
    'scanner:transformer': 'Potential prompt injection detected: transformer model',
    'scanner:vectordb': 'Potential prompt injection detected: vector similarity',
    'scanner:response-similarity': 'Potential prompt injection detected: prompt-response similarity'
}


def calculate_entropy(text) -> float:
    prob = [text.count(c) / len(text) for c in set(text)]
    entropy = -sum(p * math.log2(p) for p in prob)
    return entropy


class Manager:
    def __init__(self,
                 scanners: List[BaseScanner],
                 auto_update: bool = False,
                 update_threshold: int = 3,
                 db_client=None,
                 name: str = 'input'):
        self.name = f'dispatch:{name}'
        self.dispatcher = Scanner(scanners)
        self.auto_update = auto_update
        self.update_threshold = update_threshold
        self.db_client = db_client

        if self.auto_update:
            if self.db_client is None:
                logger.warn(f'{self.name} Auto-update disabled: db client is None')
            else:
                logger.info(f'{self.name} Auto-update vectordb enabled: threshold={self.update_threshold}')

    def perform_scan(self, prompt: str, prompt_response: str = None) -> dict:
        resp = ResponseModel(
            status='success',
            prompt=prompt,
            prompt_response=prompt_response,
            prompt_entropy=calculate_entropy(prompt),
        )

        resp.uuid = str(resp.uuid)

        if not prompt:
            resp.errors.append('Input prompt value is empty')
            resp.status = StatusEmum.FAILED
            logger.error(f'{self.name} Input prompt value is empty')
            return resp.dict()

        logger.info(f'{self.name} Dispatching scan request id={resp.uuid}')

        scan_results = self.dispatcher.run(
            prompt=prompt,
            prompt_response=prompt_response,
            scan_id=resp.uuid
        )

        total_matches = 0
        total_scanners = 0
        failed_scanners = 0

        for scanner_name, results in scan_results.items():
            total_scanners += 1

            if isinstance(results, dict) and results.get('error'):
                failed_scanners += 1
                resp.errors.append(f'Error in {scanner_name}: {results["error"]}')
                resp.results[scanner_name] = ScannerResultModel(
                    status=StatusEmum.FAILED,
                    matches=[],
                    error=results['error']
                )
            else:
                matches = results.get('matches') if isinstance(results, dict) else results
                if matches is None:
                    matches = []
                resp.results[scanner_name] = ScannerResultModel(
                    status=StatusEmum.SUCCESS,
                    matches=matches
                )
                if len(matches) > 0 and scanner_name != 'scanner:sentiment':
                    total_matches += 1

        if total_scanners > 0 and failed_scanners == total_scanners:
            resp.status = StatusEmum.FAILED
        elif failed_scanners > 0:
            resp.status = StatusEmum.PARTIAL

        for scanner_name, message in messages.items():
            scanner_result = resp.results.get(scanner_name)
            if scanner_result is not None and len(scanner_result.matches) > 0 \
                    and message not in resp.messages:
                resp.messages.append(message)

        logger.info(f'{self.name} Total scanner matches: {total_matches}')
        if self.auto_update and failed_scanners > 0:
            logger.info(
                f'{self.name} (auto-update) Skipping incomplete scan; '
                f'failed_scanners={failed_scanners} id={resp.uuid}'
            )

        if (self.auto_update and self.db_client is not None
                and failed_scanners == 0
                and total_matches >= self.update_threshold):
            logger.info(f'{self.name} (auto-update) Adding detected prompt to db id={resp.uuid}')
            doc_id = self.db_client.add_texts(
                [prompt],
                [
                    {
                        'uuid': resp.uuid,
                        'source': 'auto-update',
                        'timestamp': timestamp_str(),
                        'threshold': self.update_threshold
                    }
                ]
            )
            logger.success(f'{self.name} (auto-update) Successful doc_id={doc_id} id={resp.uuid}')

        logger.info(f'{self.name} Returning response object id={resp.uuid}')

        return resp.dict()


class Scanner:
    def __init__(self, scanners: List[BaseScanner]):
        self.name = 'dispatch:scan'
        self.scanners = scanners

    def run(self, prompt: str, scan_id: str, prompt_response: str = None) -> Dict:
        response = {}

        for scanner in self.scanners:
            scan_obj = ScanModel(
                prompt=prompt,
                prompt_response=prompt_response
            )

            try:
                logger.info(f'Running scanner: {scanner.name}; id={scan_id}')

                updated = scanner.analyze(scan_obj, scan_id)
                response[scanner.name] = ScannerResultModel(
                    status=StatusEmum.SUCCESS,
                    matches=[res.dict() for res in updated.results]
                ).dict()
                logger.success(f'Successfully ran scanner: {scanner.name} id={scan_id}')

            except Exception as err:
                logger.error(f'Failed to run scanner: {scanner.name}, Error: {str(err)} id={scan_id}')
                response[scanner.name] = ScannerResultModel(
                    status=StatusEmum.FAILED,
                    error=str(err)
                ).dict()

        return response
